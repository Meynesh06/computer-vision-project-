import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tissue_classifier.finetune import (
    LoRAClassifier,
    attach_lora,
    evaluate_lora,
    train_lora,
)


class TinyBackboneWithQKV(nn.Module):
    """A minimal backbone with a linear layer literally named 'qkv', so peft's
    name-substring target_modules matching has something real to attach to."""

    def __init__(self, embed_dim=8):
        super().__init__()
        self.qkv = nn.Linear(3 * 4 * 4, embed_dim)

    def forward(self, x):
        return self.qkv(x.flatten(1))


def test_attach_lora_freezes_base_and_leaves_adapters_trainable():
    backbone = TinyBackboneWithQKV()
    lora_model = attach_lora(backbone, r=2, lora_alpha=4, target_modules=["qkv"])

    trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in lora_model.parameters())

    assert 0 < trainable < total


def test_train_lora_fits_separable_data():
    backbone = TinyBackboneWithQKV(embed_dim=8)
    lora_model = attach_lora(backbone, r=2, lora_alpha=4, target_modules=["qkv"])
    classifier = LoRAClassifier(lora_model, embedding_dim=8, num_classes=2)

    rng = np.random.default_rng(0)
    n_per_class = 10
    images = []
    labels = []
    for class_id in range(2):
        base = np.full((3, 4, 4), fill_value=float(class_id) * 5.0, dtype="float32")
        for _ in range(n_per_class):
            images.append(base + rng.normal(scale=0.05, size=(3, 4, 4)).astype("float32"))
            labels.append(class_id)
    images = torch.from_numpy(np.stack(images))
    labels = torch.tensor(labels, dtype=torch.long)
    dataloader = DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=True)

    trained = train_lora(classifier, dataloader, epochs=20, lr=0.05, device="cpu")
    y_true, y_pred = evaluate_lora(trained, dataloader, device="cpu")

    accuracy = (y_true == y_pred).mean()
    assert accuracy > 0.9
