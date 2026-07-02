import logging
import random
import shutil
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

CLASS_NAMES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def class_name_to_id(name: str) -> int:
    return CLASS_NAMES.index(name)


def id_to_class_name(idx: int) -> str:
    return CLASS_NAMES[idx]


def _list_images(class_dir: Path) -> list[Path]:
    return sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


class TissuePatchDataset(Dataset):
    def __init__(self, root_dir, transform, class_names: list[str] = CLASS_NAMES):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.class_names = class_names
        self.samples: list[tuple[Path, int]] = []

        for name in class_names:
            class_dir = self.root_dir / name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")
            images = _list_images(class_dir)
            if not images:
                raise FileNotFoundError(f"Empty class directory: {class_dir}")
            label = class_name_to_id(name)
            self.samples.extend((path, label) for path in images)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        for attempt in range(len(self.samples)):
            path, label = self.samples[(idx + attempt) % len(self.samples)]
            try:
                image = Image.open(path).convert("RGB")
            except (UnidentifiedImageError, OSError):
                logger.warning("Skipping corrupt/unreadable image: %s", path)
                continue
            return self.transform(image), label

        raise RuntimeError("All images in the dataset are corrupt/unreadable")


def create_local_subset(source_dir, dest_dir, per_class: int, seed: int) -> None:
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    rng = random.Random(seed)

    for name in CLASS_NAMES:
        source_class_dir = source_dir / name
        images = _list_images(source_class_dir)
        if len(images) < per_class:
            raise ValueError(
                f"Class {name} has only {len(images)} images, need {per_class}"
            )
        chosen = sorted(rng.sample(images, per_class), key=lambda p: p.name)

        dest_class_dir = dest_dir / name
        dest_class_dir.mkdir(parents=True, exist_ok=True)
        for path in chosen:
            shutil.copy2(path, dest_class_dir / path.name)
