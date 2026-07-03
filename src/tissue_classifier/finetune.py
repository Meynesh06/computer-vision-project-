from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict


def _unwrap(classifier: nn.Module) -> nn.Module:
    """Reach through a DataParallel wrapper (if present) to the underlying
    LoRAClassifier, so state-dict helpers and attribute access work the same
    whether or not multi-GPU wrapping is in use."""
    return classifier.module if isinstance(classifier, nn.DataParallel) else classifier


def get_lora_state(classifier: nn.Module) -> dict:
    """Adapter-only checkpoint payload: the small LoRA deltas + classifier
    head, not the frozen multi-hundred-million-param backbone. This is what
    keeps checkpoints small enough to sync to GCS every few hundred steps."""
    module = _unwrap(classifier)
    return {
        "lora_state": get_peft_model_state_dict(module.backbone),
        "head_state": module.head.state_dict(),
    }


def load_lora_state(classifier: nn.Module, state: dict) -> None:
    module = _unwrap(classifier)
    set_peft_model_state_dict(module.backbone, state["lora_state"])
    module.head.load_state_dict(state["head_state"])


def wrap_data_parallel(classifier: nn.Module, num_gpus: int) -> nn.Module:
    """Wrap in torch.nn.DataParallel when multiple GPUs are requested and
    available; otherwise return the classifier unchanged. A no-op path when
    num_gpus <= 1 or only one CUDA device is visible."""
    if num_gpus > 1 and torch.cuda.device_count() > 1:
        return nn.DataParallel(classifier, device_ids=list(range(num_gpus)))
    return classifier


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
    start_epoch: int = 0,
    optimizer_state: dict | None = None,
    checkpoint_every_steps: int | None = None,
    on_checkpoint: Callable[[dict], None] | None = None,
    num_gpus: int = 1,
) -> LoRAClassifier:
    classifier = classifier.to(device)
    classifier = wrap_data_parallel(classifier, num_gpus)
    trainable_params = [p for p in classifier.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    loss_fn = nn.CrossEntropyLoss()

    global_step = 0
    classifier.train()
    for epoch in range(start_epoch, epochs):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = classifier(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            global_step += 1
            if (
                checkpoint_every_steps
                and on_checkpoint is not None
                and global_step % checkpoint_every_steps == 0
            ):
                on_checkpoint(
                    {
                        "epoch": epoch,
                        "step": global_step,
                        "model_state": get_lora_state(classifier),
                        "optimizer_state": optimizer.state_dict(),
                    }
                )

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
