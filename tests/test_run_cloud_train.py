from pathlib import Path

import pytest
import torch
import torch.nn as nn
from PIL import Image

from tissue_classifier.config import load_config
from tissue_classifier.data import CLASS_NAMES
from scripts.run_cloud_train import load_checkpoint, save_checkpoint

# Every key `run_cloud_train.main()` reads off `config` before/while running
# the combined probe+LoRA pipeline (including the train/eval split via
# `eval_data_dir`). Any shipped config missing one of these blows up with a
# KeyError partway through main() -- see the lora_finetune.yaml /
# linear_probe.yaml key-set bug this test guards against.
#
# `configs/local_smoke.yaml` is intentionally excluded here: it is consumed
# by `scripts/run_local_smoke.py`'s main() instead, which evaluates on the
# same dataset it trains on and therefore has no `eval_data_dir` key.
REQUIRED_CLOUD_TRAIN_CONFIG_KEYS = [
    "data_dir",
    "eval_data_dir",
    "device",
    "precision",
    "backbone",
    "batch_size",
    "seed",
    "epochs_probe",
    "lr_probe",
    "epochs_lora",
    "lr_lora",
    "lora_r",
    "lora_alpha",
    "target_modules",
    "cache_dir",
    "checkpoint_dir",
]

# Keys read by `scripts/run_local_smoke.py`'s main() -- same combined
# pipeline, but no train/eval split, so no `eval_data_dir`.
REQUIRED_LOCAL_SMOKE_CONFIG_KEYS = [
    key for key in REQUIRED_CLOUD_TRAIN_CONFIG_KEYS if key != "eval_data_dir"
]


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/linear_probe.yaml",
        "configs/lora_finetune.yaml",
    ],
)
def test_shipped_cloud_config_has_all_keys_main_needs(config_path):
    config = load_config(config_path)
    missing = [
        key for key in REQUIRED_CLOUD_TRAIN_CONFIG_KEYS if key not in config
    ]
    assert not missing, (
        f"{config_path} is missing keys read by run_cloud_train.main(): {missing}"
    )


def test_shipped_local_smoke_config_has_all_keys_main_needs():
    config = load_config("configs/local_smoke.yaml")
    missing = [
        key for key in REQUIRED_LOCAL_SMOKE_CONFIG_KEYS if key not in config
    ]
    assert not missing, (
        f"configs/local_smoke.yaml is missing keys read by "
        f"run_local_smoke.main(): {missing}"
    )


class FakeBackboneWithQKV(nn.Module):
    def __init__(self, embed_dim=8):
        super().__init__()
        self.pretrained_cfg = {
            "input_size": (3, 8, 8),
            "mean": (0.5, 0.5, 0.5),
            "std": (0.5, 0.5, 0.5),
            "crop_pct": 1.0,
            "interpolation": "bilinear",
        }
        self.qkv = nn.Linear(3 * 8 * 8, embed_dim)

    def forward(self, x):
        return self.qkv(x.flatten(1))


def fake_loader(model_name, pretrained, num_classes):
    return FakeBackboneWithQKV()


def test_save_and_load_checkpoint_roundtrip(tmp_path):
    state = {"epoch": 3, "model_state": {"a": torch.tensor([1.0, 2.0])}}

    save_checkpoint(state, tmp_path)
    loaded = load_checkpoint(tmp_path)

    assert loaded["epoch"] == 3
    assert torch.equal(loaded["model_state"]["a"], torch.tensor([1.0, 2.0]))


def test_load_checkpoint_returns_none_when_absent(tmp_path):
    assert load_checkpoint(tmp_path) is None


def _write_cloud_config(tmp_path, train_dir, eval_dir):
    config_path = tmp_path / "cloud.yaml"
    config_path.write_text(f"""
data_dir: {train_dir}
eval_data_dir: {eval_dir}
device: cpu
precision: fp32
backbone: fake/backbone
batch_size: 4
seed: 0
epochs_probe: 2
lr_probe: 0.05
epochs_lora: 1
lr_lora: 0.01
lora_r: 2
lora_alpha: 4
target_modules: ["qkv"]
checkpoint_every_steps: 1
cache_dir: {tmp_path}/cache
checkpoint_dir: {tmp_path}/checkpoints
""")
    return config_path


def _make_dataset(root):
    for name in CLASS_NAMES:
        d = root / name
        d.mkdir(parents=True)
        for i in range(4):
            Image.new("RGB", (16, 16)).save(d / f"img_{i}.tif")


def test_run_cloud_train_end_to_end_evaluates_on_eval_dir(tmp_path):
    from scripts.run_cloud_train import main

    train_dir = tmp_path / "train"
    eval_dir = tmp_path / "eval"
    _make_dataset(train_dir)
    _make_dataset(eval_dir)

    config_path = _write_cloud_config(tmp_path, train_dir, eval_dir)

    results = main(str(config_path), backbone_loader=fake_loader, fresh=True)

    assert "probe_metrics" in results
    assert "lora_metrics" in results
    checkpoint_dir = tmp_path / "checkpoints"
    assert (checkpoint_dir / "lora_checkpoint.pt").exists()
