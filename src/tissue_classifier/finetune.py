import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model


def attach_lora(
    model: nn.Module, r: int, lora_alpha: int, target_modules: list[str]
) -> nn.Module:
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        bias="none",
    )
    return get_peft_model(model, lora_config)


class LoRAClassifier(nn.Module):
    def __init__(self, lora_model: nn.Module, embedding_dim: int, num_classes: int):
        super().__init__()
        self.backbone = lora_model
        self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        embeddings = self.backbone(x)
        return self.head(embeddings)


def train_lora(
    classifier: LoRAClassifier,
    dataloader,
    epochs: int,
    lr: float,
    device: str,
) -> LoRAClassifier:
    classifier = classifier.to(device)
    trainable_params = [p for p in classifier.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    classifier.train()
    for _ in range(epochs):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = classifier(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

    classifier.eval()
    return classifier


def evaluate_lora(
    classifier: LoRAClassifier, dataloader, device: str
) -> tuple[np.ndarray, np.ndarray]:
    classifier.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = classifier(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())

    return np.concatenate(all_labels), np.concatenate(all_preds)
