import os
from pathlib import Path
from typing import Callable

import numpy as np
import timm
import torch
import torch.nn as nn
from timm.layers import SwiGLUPacked

# Registry of model_name -> extra kwargs that must be passed to
# timm.create_model for that *specific* model's own documented quirks.
# Only react to exact, known model_name strings here -- never widen this to
# a guess for unknown backbones (e.g. the prov-gigapath fallback), since we
# have no evidence those need (or don't need) the same treatment.
_MODEL_CREATE_KWARGS: dict[str, dict] = {
    # Virchow uses a SwiGLU-based MLP; timm's plain ViT MLP shape mismatches
    # the checkpoint unless these are passed explicitly (Paige's documented
    # usage: https://huggingface.co/paige-ai/Virchow).
    "hf-hub:paige-ai/Virchow": {
        "mlp_layer": SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
    },
}


class _PooledTokenBackbone(nn.Module):
    """Wraps a raw backbone whose forward() returns the unpooled per-token
    sequence `[batch, num_tokens, embed_dim]` (as Virchow's hosted config
    does, since its timm `global_pool` is disabled) and produces a single
    usable embedding per Paige's documented usage: concatenate the CLS token
    with the mean of the remaining patch tokens -> `[batch, 2 * embed_dim]`.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.pretrained_cfg = backbone.pretrained_cfg

    def forward(self, x):
        tokens = self.backbone(x)
        cls_token = tokens[:, 0]
        patch_tokens = tokens[:, 1:]
        return torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=-1)


def _default_loader(model_name: str, pretrained: bool, num_classes: int) -> torch.nn.Module:
    """Default `loader` used by load_backbone when the caller doesn't supply
    one. Applies model-specific timm kwargs from `_MODEL_CREATE_KWARGS` (and
    the matching pooling wrapper) only for exact known model_name matches;
    every other model_name goes through plain timm.create_model unchanged.
    """
    extra_kwargs = _MODEL_CREATE_KWARGS.get(model_name)
    if extra_kwargs is not None:
        backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes, **extra_kwargs
        )
        return _PooledTokenBackbone(backbone)
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)


def load_backbone(
    model_name: str, device: str, loader: Callable | None = None
) -> torch.nn.Module:
    loader = loader or _default_loader
    model = loader(model_name, pretrained=True, num_classes=0)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model.to(device)


def get_transform(model) -> Callable:
    data_config = timm.data.resolve_data_config({}, model=model)
    return timm.data.create_transform(**data_config)


def _partial_cache_path(cache_path: Path) -> Path:
    return cache_path.parent / f"{cache_path.stem}.partial.npz"


def extract_embeddings(
    model,
    dataloader,
    device: str,
    cache_path: Path | None = None,
    checkpoint_every_batches: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract embeddings for every item in dataloader. If cache_path is
    given, periodically flushes progress to a sibling `*.partial.npz` file
    (atomically) so a preempted run can resume from the last completed
    batch instead of starting over -- this is likely a multi-hour phase for
    the full dataset. The partial file is removed once the final cache is
    written."""
    if cache_path is not None and cache_path.exists():
        cached = np.load(cache_path)
        return cached["embeddings"], cached["labels"]

    partial_path = _partial_cache_path(cache_path) if cache_path is not None else None

    all_embeddings = []
    all_labels = []
    completed_batches = 0
    if partial_path is not None and partial_path.exists():
        partial = np.load(partial_path)
        all_embeddings = [partial["embeddings"]]
        all_labels = [partial["labels"]]
        completed_batches = int(partial["completed_batches"])

    def _flush_partial() -> None:
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = partial_path.parent / f"{partial_path.stem}.tmp.npz"
        np.savez(
            str(tmp_path),
            embeddings=np.concatenate(all_embeddings, axis=0),
            labels=np.concatenate(all_labels, axis=0),
            completed_batches=completed_batches,
        )
        os.replace(str(tmp_path), str(partial_path))

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(dataloader):
            if batch_index < completed_batches:
                continue
            images = images.to(device)
            embeddings = model(images)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())
            completed_batches = batch_index + 1

            if partial_path is not None and completed_batches % checkpoint_every_batches == 0:
                _flush_partial()

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.parent / f"{cache_path.stem}.tmp.npz"
        np.savez(str(tmp_path), embeddings=embeddings, labels=labels)
        os.replace(str(tmp_path), str(cache_path))
        if partial_path is not None and partial_path.exists():
            partial_path.unlink()

    return embeddings, labels
