import sys
from pathlib import Path
from typing import Callable

from torch.utils.data import DataLoader

from tissue_classifier.config import load_config, resolve_device, resolve_precision
from tissue_classifier.data import CLASS_NAMES, TissuePatchDataset
from tissue_classifier.finetune import (
    LoRAClassifier,
    attach_lora,
    evaluate_lora,
    train_lora,
)
from tissue_classifier.foundation_model import (
    extract_embeddings,
    get_transform,
    load_backbone,
)
from tissue_classifier.metrics import compute_metrics
from tissue_classifier.probe import evaluate_probe, train_probe


def main(config_path: str, backbone_loader: Callable | None = None) -> dict:
    config = load_config(config_path)
    device = resolve_device(config["device"])
    resolve_precision(config["precision"], device)  # validated, not yet applied at this scale

    backbone = load_backbone(config["backbone"], device, loader=backbone_loader)
    transform = get_transform(backbone)

    dataset = TissuePatchDataset(config["data_dir"], transform=transform)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)

    cache_path = Path(config["cache_dir"]) / "local_smoke_embeddings.npz"
    embeddings, labels = extract_embeddings(backbone, dataloader, device, cache_path)

    probe = train_probe(
        embeddings,
        labels,
        num_classes=len(CLASS_NAMES),
        epochs=config["epochs_probe"],
        lr=config["lr_probe"],
        device=device,
    )
    y_true, y_pred = evaluate_probe(probe, embeddings, labels, device)
    probe_metrics = compute_metrics(y_true.tolist(), y_pred.tolist(), CLASS_NAMES)

    lora_backbone = load_backbone(config["backbone"], device, loader=backbone_loader)
    lora_model = attach_lora(
        lora_backbone,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["target_modules"],
    )
    classifier = LoRAClassifier(
        lora_model, embedding_dim=embeddings.shape[1], num_classes=len(CLASS_NAMES)
    )
    shuffled_dataloader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True
    )
    trained_classifier = train_lora(
        classifier,
        shuffled_dataloader,
        epochs=config["epochs_lora"],
        lr=config["lr_lora"],
        device=device,
    )
    lora_y_true, lora_y_pred = evaluate_lora(trained_classifier, dataloader, device)
    lora_metrics = compute_metrics(
        lora_y_true.tolist(), lora_y_pred.tolist(), CLASS_NAMES
    )

    return {"probe_metrics": probe_metrics, "lora_metrics": lora_metrics}


if __name__ == "__main__":
    results = main(sys.argv[1])
    print(results)
