import os
from pathlib import Path
from typing import Callable

import numpy as np
import timm
import torch


def load_backbone(
    model_name: str, device: str, loader: Callable | None = None
) -> torch.nn.Module:
    loader = loader or timm.create_model
    model = loader(model_name, pretrained=True, num_classes=0)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model.to(device)


def get_transform(model) -> Callable:
    data_config = timm.data.resolve_data_config({}, model=model)
    return timm.data.create_transform(**data_config)


def extract_embeddings(
    model, dataloader, device: str, cache_path: Path | None = None
) -> tuple[np.ndarray, np.ndarray]:
    if cache_path is not None and cache_path.exists():
        cached = np.load(cache_path)
        return cached["embeddings"], cached["labels"]

    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            embeddings = model(images)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.parent / f"{cache_path.stem}.tmp.npz"
        np.savez(str(tmp_path), embeddings=embeddings, labels=labels)
        os.replace(str(tmp_path), str(cache_path))

    return embeddings, labels
