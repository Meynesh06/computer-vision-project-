from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def compute_metrics(
    y_true: list[int], y_pred: list[int], class_names: list[str]
) -> dict:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    labels = list(range(len(class_names)))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class_f1": {
            name: float(score) for name, score in zip(class_names, per_class)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
