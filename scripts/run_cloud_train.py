import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader

from tissue_classifier.config import load_config, resolve_device, resolve_precision
from tissue_classifier.data import CLASS_NAMES, TissuePatchDataset
from tissue_classifier.finetune import (
    LoRAClassifier,
    attach_lora,
    evaluate_lora,
    get_lora_state,
    load_lora_state,
    train_lora,
)
from tissue_classifier.foundation_model import (
    extract_embeddings,
    get_transform,
    load_backbone,
)
from tissue_classifier.metrics import compute_metrics
from tissue_classifier.probe import evaluate_probe, train_probe

CHECKPOINT_FILENAME = "lora_checkpoint.pt"
RESULTS_FILENAME = "results.json"


def save_checkpoint(state: dict, checkpoint_dir: str | Path) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_path = checkpoint_dir / CHECKPOINT_FILENAME
    tmp_path = final_path.with_suffix(".pt.tmp")
    torch.save(state, tmp_path)
    os.replace(tmp_path, final_path)


def load_checkpoint(checkpoint_dir: str | Path) -> dict | None:
    checkpoint_path = Path(checkpoint_dir) / CHECKPOINT_FILENAME
    if not checkpoint_path.exists():
        return None
    return torch.load(checkpoint_path, weights_only=True)


def _gcs_uri(bucket: str, filename: str) -> str:
    return f"{bucket.rstrip('/')}/{filename}"


def sync_checkpoint_to_gcs(local_path: str | Path, bucket: str) -> None:
    """Best-effort push of the (now adapter-only, so small) checkpoint to
    GCS. Fails soft: a transient network/gsutil error shouldn't kill a
    training run -- the monitor script is responsible for surfacing repeated
    sync failures."""
    try:
        subprocess.run(
            ["gsutil", "cp", str(local_path), _gcs_uri(bucket, CHECKPOINT_FILENAME)],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"warning: checkpoint GCS sync failed: {exc}", file=sys.stderr)


def maybe_pull_checkpoint_from_gcs(checkpoint_dir: str | Path, bucket: str | None) -> None:
    """Pull the latest checkpoint down from GCS if no local checkpoint exists
    yet -- lets a brand-new/replacement VM resume a preempted run."""
    if not bucket:
        return
    local_path = Path(checkpoint_dir) / CHECKPOINT_FILENAME
    if local_path.exists():
        return
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["gsutil", "cp", _gcs_uri(bucket, CHECKPOINT_FILENAME), str(local_path)],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"warning: checkpoint GCS pull failed: {exc}", file=sys.stderr)


def sync_results_to_gcs(results: dict, bucket: str) -> None:
    try:
        subprocess.run(
            ["gsutil", "cp", "-", _gcs_uri(bucket, RESULTS_FILENAME)],
            input=json.dumps(results).encode(),
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"warning: results GCS sync failed: {exc}", file=sys.stderr)


def main(
    config_path: str,
    backbone_loader: Callable | None = None,
    fresh: bool = False,
) -> dict:
    config = load_config(config_path)
    torch.manual_seed(config["seed"])
    device = resolve_device(config["device"])
    resolve_precision(config["precision"], device)  # validated, not yet applied at this scale

    gcs_bucket = config.get("gcs_checkpoint_bucket")
    num_gpus = config.get("num_gpus", 1)

    backbone = load_backbone(config["backbone"], device, loader=backbone_loader)
    transform = get_transform(backbone)

    train_dataset = TissuePatchDataset(config["data_dir"], transform=transform)
    eval_dataset = TissuePatchDataset(config["eval_data_dir"], transform=transform)
    train_dataloader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=False
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=config["batch_size"], shuffle=False
    )

    cache_dir = Path(config["cache_dir"])
    train_embeddings, train_labels = extract_embeddings(
        backbone, train_dataloader, device, cache_dir / "train_embeddings.npz"
    )
    eval_embeddings, eval_labels = extract_embeddings(
        backbone, eval_dataloader, device, cache_dir / "eval_embeddings.npz"
    )

    probe = train_probe(
        train_embeddings,
        train_labels,
        num_classes=len(CLASS_NAMES),
        epochs=config["epochs_probe"],
        lr=config["lr_probe"],
        device=device,
    )
    probe_y_true, probe_y_pred = evaluate_probe(
        probe, eval_embeddings, eval_labels, device
    )
    probe_metrics = compute_metrics(
        probe_y_true.tolist(), probe_y_pred.tolist(), CLASS_NAMES
    )

    checkpoint_dir = config["checkpoint_dir"]
    if not fresh:
        maybe_pull_checkpoint_from_gcs(checkpoint_dir, gcs_bucket)
    checkpoint = None if fresh else load_checkpoint(checkpoint_dir)

    lora_backbone = load_backbone(config["backbone"], device, loader=backbone_loader)
    lora_model = attach_lora(
        lora_backbone,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["target_modules"],
    )
    classifier = LoRAClassifier(
        lora_model,
        embedding_dim=train_embeddings.shape[1],
        num_classes=len(CLASS_NAMES),
    )
    start_epoch = 0
    optimizer_state = None
    if checkpoint is not None:
        load_lora_state(classifier, checkpoint["model_state"])
        start_epoch = checkpoint["epoch"]
        optimizer_state = checkpoint.get("optimizer_state")

    def on_checkpoint(state: dict) -> None:
        save_checkpoint(state, checkpoint_dir)
        if gcs_bucket:
            sync_checkpoint_to_gcs(Path(checkpoint_dir) / CHECKPOINT_FILENAME, gcs_bucket)

    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(config["seed"])
    shuffled_train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=shuffle_generator,
    )
    trained_classifier = train_lora(
        classifier,
        shuffled_train_dataloader,
        epochs=config["epochs_lora"],
        lr=config["lr_lora"],
        device=device,
        start_epoch=start_epoch,
        optimizer_state=optimizer_state,
        checkpoint_every_steps=config.get("checkpoint_every_steps"),
        on_checkpoint=on_checkpoint,
        num_gpus=num_gpus,
    )
    on_checkpoint(
        {"epoch": config["epochs_lora"], "model_state": get_lora_state(trained_classifier)}
    )

    lora_y_true, lora_y_pred = evaluate_lora(
        trained_classifier, eval_dataloader, device
    )
    lora_metrics = compute_metrics(
        lora_y_true.tolist(), lora_y_pred.tolist(), CLASS_NAMES
    )

    results = {"probe_metrics": probe_metrics, "lora_metrics": lora_metrics}
    if gcs_bucket:
        sync_results_to_gcs(results, gcs_bucket)
    return results


if __name__ == "__main__":
    fresh_flag = "--fresh" in sys.argv
    results = main(sys.argv[1], fresh=fresh_flag)
    print(results)
