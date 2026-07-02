import numpy as np

from tissue_classifier.metrics import compute_metrics
from tissue_classifier.probe import LinearProbe, evaluate_probe, train_probe


def _linearly_separable_data(n_per_class=20, num_classes=3, embedding_dim=4, seed=0):
    rng = np.random.default_rng(seed)
    embeddings = []
    labels = []
    for class_id in range(num_classes):
        center = np.zeros(embedding_dim)
        center[class_id] = 10.0  # well-separated clusters
        cluster = center + rng.normal(scale=0.1, size=(n_per_class, embedding_dim))
        embeddings.append(cluster)
        labels.extend([class_id] * n_per_class)
    return (
        np.concatenate(embeddings).astype("float32"),
        np.array(labels, dtype="int64"),
    )


def test_linear_probe_output_shape():
    probe = LinearProbe(embedding_dim=16, num_classes=9)
    import torch

    logits = probe(torch.rand(5, 16))
    assert logits.shape == (5, 9)


def test_train_probe_fits_separable_data():
    embeddings, labels = _linearly_separable_data()

    probe = train_probe(
        embeddings, labels, num_classes=3, epochs=50, lr=0.1, device="cpu"
    )
    y_true, y_pred = evaluate_probe(probe, embeddings, labels, device="cpu")
    metrics = compute_metrics(y_true.tolist(), y_pred.tolist(), ["a", "b", "c"])

    assert metrics["accuracy"] > 0.95
