import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tissue_classifier.finetune import (
    LoRAClassifier,
    attach_lora,
    evaluate_lora,
    get_lora_state,
    load_lora_state,
    train_lora,
    wrap_data_parallel,
)


class TinyBackboneWithQKV(nn.Module):
    """A minimal backbone with a linear layer literally named 'qkv', so peft's
    name-substring target_modules matching has something real to attach to."""

    def __init__(self, embed_dim=8):
        super().__init__()
        self.qkv = nn.Linear(3 * 4 * 4, embed_dim)

    def forward(self, x):
        return self.qkv(x.flatten(1))


def test_attach_lora_freezes_base_and_leaves_adapters_trainable():
    backbone = TinyBackboneWithQKV()
    lora_model = attach_lora(backbone, r=2, lora_alpha=4, target_modules=["qkv"])

    trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in lora_model.parameters())

    assert 0 < trainable < total


def test_train_lora_fits_separable_data():
    backbone = TinyBackboneWithQKV(embed_dim=8)
    lora_model = attach_lora(backbone, r=2, lora_alpha=4, target_modules=["qkv"])
    classifier = LoRAClassifier(lora_model, embedding_dim=8, num_classes=2)

    rng = np.random.default_rng(0)
    n_per_class = 10
    images = []
    labels = []
    for class_id in range(2):
        base = np.full((3, 4, 4), fill_value=float(class_id) * 5.0, dtype="float32")
        for _ in range(n_per_class):
            images.append(base + rng.normal(scale=0.05, size=(3, 4, 4)).astype("float32"))
            labels.append(class_id)
    images = torch.from_numpy(np.stack(images))
    labels = torch.tensor(labels, dtype=torch.long)
    dataloader = DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=True)

    trained = train_lora(classifier, dataloader, epochs=20, lr=0.05, device="cpu")
    y_true, y_pred = evaluate_lora(trained, dataloader, device="cpu")

    accuracy = (y_true == y_pred).mean()
    assert accuracy > 0.9


def _make_classifier():
    backbone = TinyBackboneWithQKV(embed_dim=8)
    lora_model = attach_lora(backbone, r=2, lora_alpha=4, target_modules=["qkv"])
    return LoRAClassifier(lora_model, embedding_dim=8, num_classes=2)


def _make_separable_dataloader(seed=0):
    rng = np.random.default_rng(seed)
    n_per_class = 10
    images = []
    labels = []
    for class_id in range(2):
        base = np.full((3, 4, 4), fill_value=float(class_id) * 5.0, dtype="float32")
        for _ in range(n_per_class):
            images.append(base + rng.normal(scale=0.05, size=(3, 4, 4)).astype("float32"))
            labels.append(class_id)
    images = torch.from_numpy(np.stack(images))
    labels = torch.tensor(labels, dtype=torch.long)
    return DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=False)


def test_get_lora_state_contains_only_adapter_and_head_keys():
    """Regression test for the checkpoint-scope bug: a checkpoint must only
    contain the small LoRA adapter deltas + head, never the frozen backbone's
    own weights (which would make checkpoints multi-GB at full Virchow
    scale)."""
    classifier = _make_classifier()
    state = get_lora_state(classifier)

    assert any("lora_A" in key for key in state["lora_state"])
    assert any("lora_B" in key for key in state["lora_state"])
    assert not any("base_layer" in key for key in state["lora_state"])
    assert set(state["head_state"]) == {"weight", "bias"}


def test_load_lora_state_roundtrip_restores_weights():
    classifier = _make_classifier()
    dataloader = _make_separable_dataloader()
    trained = train_lora(classifier, dataloader, epochs=5, lr=0.05, device="cpu")
    state = get_lora_state(trained)

    fresh = _make_classifier()
    load_lora_state(fresh, state)

    restored_state = get_lora_state(fresh)
    for key in state["lora_state"]:
        assert torch.equal(state["lora_state"][key], restored_state["lora_state"][key])
    for key in state["head_state"]:
        assert torch.equal(state["head_state"][key], restored_state["head_state"][key])


def test_on_checkpoint_fires_at_expected_step_cadence():
    classifier = _make_classifier()
    dataloader = _make_separable_dataloader()  # 20 samples, batch_size=4 -> 5 steps/epoch

    seen_steps = []
    train_lora(
        classifier,
        dataloader,
        epochs=2,
        lr=0.05,
        device="cpu",
        checkpoint_every_steps=3,
        on_checkpoint=lambda state: seen_steps.append(state["step"]),
    )

    # 2 epochs * 5 steps/epoch = 10 steps total; every-3rd-step -> steps 3, 6, 9
    assert seen_steps == [3, 6, 9]


def test_on_checkpoint_payload_has_expected_keys():
    classifier = _make_classifier()
    dataloader = _make_separable_dataloader()

    captured = {}
    train_lora(
        classifier,
        dataloader,
        epochs=1,
        lr=0.05,
        device="cpu",
        checkpoint_every_steps=1,
        on_checkpoint=lambda state: captured.update(state),
    )

    assert set(captured) == {"epoch", "step", "model_state", "optimizer_state"}
    assert "lora_state" in captured["model_state"]
    assert "head_state" in captured["model_state"]


def test_start_epoch_skips_completed_epochs():
    """A resume must not redo epochs already completed -- this is the
    regression test for the resume-count bug: train_lora(start_epoch=N)
    should only run epochs [N, epochs)."""
    classifier = _make_classifier()
    dataloader = _make_separable_dataloader()

    seen_epochs = []
    train_lora(
        classifier,
        dataloader,
        epochs=5,
        lr=0.05,
        device="cpu",
        start_epoch=3,
        checkpoint_every_steps=1,
        on_checkpoint=lambda state: seen_epochs.append(state["epoch"]),
    )

    # Only epochs 3 and 4 should have run (5 steps/epoch each -> 10 checkpoints,
    # all tagged with epoch 3 or 4).
    assert set(seen_epochs) == {3, 4}


def test_resume_with_optimizer_state_continues_training_equivalently():
    """Training for 10 epochs uninterrupted should reach comparable loss to
    training 5 epochs, capturing state, then resuming for 5 more with the
    saved optimizer state."""
    torch.manual_seed(0)
    classifier_full = _make_classifier()
    dataloader_full = _make_separable_dataloader()
    trained_full = train_lora(classifier_full, dataloader_full, epochs=10, lr=0.05, device="cpu")
    y_true, y_pred = evaluate_lora(trained_full, dataloader_full, device="cpu")
    full_accuracy = (y_true == y_pred).mean()

    torch.manual_seed(0)
    classifier_partial = _make_classifier()
    dataloader_partial = _make_separable_dataloader()
    captured = {}
    train_lora(
        classifier_partial,
        dataloader_partial,
        epochs=5,
        lr=0.05,
        device="cpu",
        checkpoint_every_steps=1,
        on_checkpoint=lambda state: captured.update(state),
    )
    resumed_classifier = _make_classifier()
    load_lora_state(resumed_classifier, captured["model_state"])
    resumed = train_lora(
        resumed_classifier,
        dataloader_partial,
        epochs=10,
        lr=0.05,
        device="cpu",
        start_epoch=captured["epoch"] + 1,
        optimizer_state=captured["optimizer_state"],
    )
    y_true, y_pred = evaluate_lora(resumed, dataloader_partial, device="cpu")
    resumed_accuracy = (y_true == y_pred).mean()

    assert resumed_accuracy > 0.9
    assert full_accuracy > 0.9


class _FakeDataParallel(nn.Module):
    """Stands in for torch.nn.DataParallel in tests: real DataParallel's
    constructor probes actual CUDA device properties even when given
    explicit device_ids, which crashes on a machine whose torch build has no
    CUDA support at all (not just "no GPU available"). This fake preserves
    the one thing our code depends on -- wrapping via `.module` -- without
    touching CUDA."""

    def __init__(self, module, device_ids=None):
        super().__init__()
        self.module = module
        self.device_ids = device_ids

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def test_wrap_data_parallel_wraps_when_multiple_gpus_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(nn, "DataParallel", _FakeDataParallel)
    classifier = _make_classifier()

    wrapped = wrap_data_parallel(classifier, num_gpus=2)

    assert isinstance(wrapped, _FakeDataParallel)
    assert wrapped.module is classifier


def test_wrap_data_parallel_is_noop_with_one_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    classifier = _make_classifier()

    wrapped = wrap_data_parallel(classifier, num_gpus=1)

    assert wrapped is classifier


def test_get_lora_state_unwraps_data_parallel(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(nn, "DataParallel", _FakeDataParallel)
    classifier = _make_classifier()
    wrapped = wrap_data_parallel(classifier, num_gpus=2)

    state = get_lora_state(wrapped)

    assert any("lora_A" in key for key in state["lora_state"])
    assert set(state["head_state"]) == {"weight", "bias"}
