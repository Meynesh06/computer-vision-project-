import logging

import pytest
from PIL import Image

from tissue_classifier.data import (
    CLASS_NAMES,
    TissuePatchDataset,
    class_name_to_id,
    create_local_subset,
    id_to_class_name,
)


def _make_tiny_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(path)


def _make_full_dataset(root, images_per_class=2):
    for name in CLASS_NAMES:
        for i in range(images_per_class):
            _make_tiny_image(root / name / f"img_{i}.tif")


def test_class_name_id_roundtrip():
    assert class_name_to_id("ADI") == 0
    assert class_name_to_id("TUM") == 8
    assert id_to_class_name(0) == "ADI"
    assert id_to_class_name(8) == "TUM"


def test_dataset_loads_all_classes(tmp_path):
    _make_full_dataset(tmp_path, images_per_class=2)

    dataset = TissuePatchDataset(tmp_path, transform=lambda img: img)

    assert len(dataset) == 2 * len(CLASS_NAMES)
    image, label = dataset[0]
    assert isinstance(label, int)
    assert 0 <= label < len(CLASS_NAMES)


def test_dataset_raises_on_missing_class_dir(tmp_path):
    _make_full_dataset(tmp_path, images_per_class=1)
    # remove one class directory entirely
    for f in (tmp_path / "TUM").iterdir():
        f.unlink()
    (tmp_path / "TUM").rmdir()

    with pytest.raises(FileNotFoundError, match="TUM"):
        TissuePatchDataset(tmp_path, transform=lambda img: img)


def test_dataset_raises_on_empty_class_dir(tmp_path):
    _make_full_dataset(tmp_path, images_per_class=1)
    for f in (tmp_path / "MUC").iterdir():
        f.unlink()

    with pytest.raises(FileNotFoundError, match="MUC"):
        TissuePatchDataset(tmp_path, transform=lambda img: img)


def test_dataset_skips_corrupt_file_and_logs(tmp_path, caplog):
    _make_full_dataset(tmp_path, images_per_class=2)
    # corrupt one file in the first class
    corrupt_path = tmp_path / CLASS_NAMES[0] / "img_0.tif"
    corrupt_path.write_bytes(b"not an image")

    dataset = TissuePatchDataset(tmp_path, transform=lambda img: img)

    with caplog.at_level(logging.WARNING):
        image, label = dataset[0]  # first item is the corrupt one; should skip to next

    assert label == 0  # still within class 0 (second image in ADI)
    assert "corrupt" in caplog.text.lower() or "unreadable" in caplog.text.lower()


def test_create_local_subset_is_deterministic(tmp_path):
    source = tmp_path / "source"
    dest_a = tmp_path / "dest_a"
    dest_b = tmp_path / "dest_b"
    _make_full_dataset(source, images_per_class=10)

    create_local_subset(source, dest_a, per_class=3, seed=42)
    create_local_subset(source, dest_b, per_class=3, seed=42)

    for name in CLASS_NAMES:
        files_a = sorted(p.name for p in (dest_a / name).iterdir())
        files_b = sorted(p.name for p in (dest_b / name).iterdir())
        assert files_a == files_b
        assert len(files_a) == 3
