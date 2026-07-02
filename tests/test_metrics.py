from tissue_classifier.metrics import compute_metrics


def test_perfect_predictions_score_1():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2]
    class_names = ["ADI", "BACK", "DEB"]

    result = compute_metrics(y_true, y_pred, class_names)

    assert result["accuracy"] == 1.0
    assert result["per_class_f1"] == {"ADI": 1.0, "BACK": 1.0, "DEB": 1.0}
    assert result["confusion_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


def test_all_wrong_predictions_score_0():
    y_true = [0, 0]
    y_pred = [1, 1]
    class_names = ["ADI", "BACK"]

    result = compute_metrics(y_true, y_pred, class_names)

    assert result["accuracy"] == 0.0
    assert result["per_class_f1"]["ADI"] == 0.0


def test_mismatched_lengths_raises():
    import pytest

    with pytest.raises(ValueError, match="same length"):
        compute_metrics([0, 1], [0], ["ADI", "BACK"])
