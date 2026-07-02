import numpy as np
import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


def train_probe(
    embeddings: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    epochs: int,
    lr: float,
    device: str,
) -> LinearProbe:
    embedding_dim = embeddings.shape[1]
    probe = LinearProbe(embedding_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    x = torch.from_numpy(embeddings).to(device)
    y = torch.from_numpy(labels).long().to(device)

    probe.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = probe(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

    probe.eval()
    return probe


def evaluate_probe(
    probe: LinearProbe, embeddings: np.ndarray, labels: np.ndarray, device: str
) -> tuple[np.ndarray, np.ndarray]:
    x = torch.from_numpy(embeddings).to(device)
    probe.eval()
    with torch.no_grad():
        logits = probe(x)
        preds = logits.argmax(dim=1).cpu().numpy()
    return labels, preds
