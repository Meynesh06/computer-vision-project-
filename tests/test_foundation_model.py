import numpy as np
import torch
import torch.nn as nn
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
