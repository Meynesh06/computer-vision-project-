import numpy as np
import pytest
import torch
import torch.nn as nn
from timm.layers import SwiGLUPacked
from torch.utils.data import DataLoader, TensorDataset

from tissue_classifier.foundation_model import (
    extract_embeddings,
    get_transform,
    load_backbone,
)


class FakeBackbone(nn.Module):
    """Mimics enough of a timm model's interface for our code to exercise real logic
    without downloading real weights."""

    def __init__(self, embed_dim=16):
        super().__init__()
        self.pretrained_cfg = {
            "input_size": (3, 8, 8),
            "mean": (0.5, 0.5, 0.5),
            "std": (0.5, 0.5, 0.5),
            "crop_pct": 1.0,
            "interpolation": "bilinear",
        }
        self.linear = nn.Linear(3 * 8 * 8, embed_dim)

    def forward(self, x):
        return self.linear(x.flatten(1))


def fake_loader(model_name, pretrained, num_classes):
    return FakeBackbone()


def test_load_backbone_is_frozen_and_in_eval_mode():
    model = load_backbone("fake/backbone", device="cpu", loader=fake_loader)

    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())


def test_get_transform_returns_callable_matching_input_size():
    model = load_backbone("fake/backbone", device="cpu", loader=fake_loader)
    transform = get_transform(model)

    from PIL import Image

    image = Image.new("RGB", (32, 32))
    tensor = transform(image)

    assert tensor.shape == (3, 8, 8)


def test_extract_embeddings_shapes_and_caches(tmp_path):
    model = load_backbone("fake/backbone", device="cpu", loader=fake_loader)
    images = torch.rand(5, 3, 8, 8)
    labels = torch.tensor([0, 1, 0, 1, 2])
    dataloader = DataLoader(TensorDataset(images, labels), batch_size=2)
    cache_path = tmp_path / "embeddings.npz"

    embeddings, out_labels = extract_embeddings(
        model, dataloader, device="cpu", cache_path=cache_path
    )

    assert embeddings.shape == (5, 16)
    assert out_labels.tolist() == [0, 1, 0, 1, 2]
    assert cache_path.exists()


def test_extract_embeddings_uses_cache_on_second_call(tmp_path):
    model = load_backbone("fake/backbone", device="cpu", loader=fake_loader)
    images = torch.rand(3, 3, 8, 8)
    labels = torch.tensor([0, 1, 2])
    dataloader = DataLoader(TensorDataset(images, labels), batch_size=2)
    cache_path = tmp_path / "embeddings.npz"

    first_embeddings, _ = extract_embeddings(
        model, dataloader, device="cpu", cache_path=cache_path
    )
    # mutate the cache to prove the second call reads it instead of recomputing
    saved = np.load(cache_path)
    np.savez(cache_path, embeddings=saved["embeddings"] * 0, labels=saved["labels"])

    second_embeddings, _ = extract_embeddings(
        model, dataloader, device="cpu", cache_path=cache_path
    )

    assert np.allclose(second_embeddings, 0)


class _CountingModel(nn.Module):
    """Wraps a model to count forward() calls, so a resume test can verify
    already-completed batches are not recomputed."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.call_count = 0

    def forward(self, x):
        self.call_count += 1
        return self.inner(x)


class _InterruptingDataLoader:
    """Wraps a real dataloader and raises partway through iteration, to
    simulate a preemption mid-embedding-extraction."""

    def __init__(self, real_loader, fail_after):
        self._real = real_loader
        self._fail_after = fail_after

    def __iter__(self):
        for i, batch in enumerate(self._real):
            if i >= self._fail_after:
                raise RuntimeError("simulated interruption")
            yield batch


def test_extract_embeddings_resumes_without_recomputing_completed_batches(tmp_path):
    model = load_backbone("fake/backbone", device="cpu", loader=fake_loader)
    images = torch.rand(6, 3, 8, 8)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    dataloader = DataLoader(TensorDataset(images, labels), batch_size=2)  # 3 batches
    cache_path = tmp_path / "embeddings.npz"
    partial_path = tmp_path / "embeddings.partial.npz"

    interrupting = _InterruptingDataLoader(dataloader, fail_after=2)
    with pytest.raises(RuntimeError):
        extract_embeddings(
            model,
            interrupting,
            device="cpu",
            cache_path=cache_path,
            checkpoint_every_batches=1,
        )

    assert partial_path.exists()
    assert not cache_path.exists()

    counting_model = _CountingModel(model)
    embeddings, out_labels = extract_embeddings(
        counting_model,
        dataloader,
        device="cpu",
        cache_path=cache_path,
        checkpoint_every_batches=1,
    )

    assert counting_model.call_count == 1  # only the 3rd (unfinished) batch recomputed
    assert embeddings.shape == (6, 16)
    assert out_labels.tolist() == [0, 1, 2, 0, 1, 2]
    assert cache_path.exists()
    assert not partial_path.exists()


class FakeRawTokenBackbone(nn.Module):
    """Mimics Virchow's *unpooled* forward output: a raw per-token sequence
    `[batch, num_tokens, embed_dim]` (1 CLS token + patch tokens), rather than
    a pooled `[batch, embed_dim]` embedding -- this is what timm.create_model
    actually returns for the real Virchow checkpoint before any pooling logic
    is applied."""

    def __init__(self, embed_dim=4, num_tokens=5):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        self.pretrained_cfg = {
            "input_size": (3, 8, 8),
            "mean": (0.5, 0.5, 0.5),
            "std": (0.5, 0.5, 0.5),
            "crop_pct": 1.0,
            "interpolation": "bilinear",
        }
        self.linear = nn.Linear(3 * 8 * 8, embed_dim * num_tokens)

    def forward(self, x):
        batch = x.shape[0]
        out = self.linear(x.flatten(1))
        return out.view(batch, self.num_tokens, self.embed_dim)


def test_load_backbone_default_path_applies_virchow_swiglu_kwargs(monkeypatch):
    """Bug 1: Virchow needs mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU
    passed to timm.create_model, or the state_dict fails to load (shape
    mismatch in the MLP fc2 weight)."""
    captured_kwargs = {}

    def fake_create_model(model_name, pretrained, num_classes, **kwargs):
        captured_kwargs.update(kwargs)
        return FakeRawTokenBackbone()

    monkeypatch.setattr(
        "tissue_classifier.foundation_model.timm.create_model", fake_create_model
    )

    load_backbone("hf-hub:paige-ai/Virchow", device="cpu")

    assert captured_kwargs.get("mlp_layer") is SwiGLUPacked
    assert captured_kwargs.get("act_layer") is torch.nn.SiLU


def test_load_backbone_default_path_pools_virchow_tokens_into_embedding(monkeypatch):
    """Bug 2: Virchow's forward() returns the raw unpooled token sequence, not
    a usable embedding. load_backbone's default path must wrap it so the
    final embedding is cat([cls_token, patch_tokens.mean(dim=1)], dim=-1)."""

    def fake_create_model(model_name, pretrained, num_classes, **kwargs):
        return FakeRawTokenBackbone(embed_dim=4, num_tokens=5)

    monkeypatch.setattr(
        "tissue_classifier.foundation_model.timm.create_model", fake_create_model
    )

    model = load_backbone("hf-hub:paige-ai/Virchow", device="cpu")
    images = torch.rand(3, 3, 8, 8)

    # Independently obtain the fake backbone's raw, unpooled per-token output
    # (the same deterministic nn.Linear forward pass the wrapper itself would
    # call) so we can compute the expected pooled embedding ourselves, rather
    # than trusting the production code's own slicing/pooling logic.
    raw_tokens = model.backbone(images)  # [batch, num_tokens=5, embed_dim=4]
    expected_cls_token = raw_tokens[:, 0]  # first token is CLS
    expected_patch_mean = raw_tokens[:, 1:].mean(dim=1)  # mean of the remaining patch tokens
    expected_embeddings = torch.cat([expected_cls_token, expected_patch_mean], dim=-1)

    embeddings = model(images)

    assert embeddings.shape == (3, 8)  # 2 * embed_dim (cls concat mean-pooled patches)
    torch.testing.assert_close(embeddings, expected_embeddings)


def test_load_backbone_default_path_non_virchow_model_gets_no_extra_kwargs(
    monkeypatch,
):
    """No regression: any model_name not in the Virchow-specific registry must
    still get timm.create_model called with today's plain kwargs, no
    SwiGLU/pooling logic applied."""
    captured_kwargs = {}

    def fake_create_model(model_name, pretrained, num_classes, **kwargs):
        captured_kwargs.update(kwargs)
        return FakeBackbone()

    monkeypatch.setattr(
        "tissue_classifier.foundation_model.timm.create_model", fake_create_model
    )

    model = load_backbone("fake/backbone", device="cpu")

    assert captured_kwargs == {}
    assert isinstance(model, FakeBackbone)


def test_explicit_loader_bypasses_default_virchow_logic(monkeypatch):
    """The loader parameter's existing contract must not change: when a
    caller passes an explicit loader, timm.create_model must not be touched
    at all, even for the Virchow model_name."""

    def explode(*args, **kwargs):
        raise AssertionError(
            "timm.create_model should not be called when an explicit loader is provided"
        )

    monkeypatch.setattr(
        "tissue_classifier.foundation_model.timm.create_model", explode
    )

    model = load_backbone("hf-hub:paige-ai/Virchow", device="cpu", loader=fake_loader)

    assert isinstance(model, FakeBackbone)
