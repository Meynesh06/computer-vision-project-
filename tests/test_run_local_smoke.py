from pathlib import Path

import torch.nn as nn
from PIL import Image

from tissue_classifier.data import CLASS_NAMES


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


def _write_smoke_config(tmp_path, data_dir):
    config_path = tmp_path / "local_smoke.yaml"
    config_path.write_text(f"""
data_dir: {data_dir}
device: cpu
precision: fp32
backbone: fake/backbone
batch_size: 4
seed: 0
epochs_probe: 3
lr_probe: 0.05
epochs_lora: 2
lr_lora: 0.01
lora_r: 2
lora_alpha: 4
target_modules: ["qkv"]
cache_dir: {tmp_path}/cache
checkpoint_dir: {tmp_path}/checkpoints
""")
    return config_path


def test_run_local_smoke_end_to_end(tmp_path):
    from scripts.run_local_smoke import main

    data_dir = tmp_path / "data"
    for name in CLASS_NAMES:
        d = data_dir / name
        d.mkdir(parents=True)
        for i in range(4):
            Image.new("RGB", (16, 16)).save(d / f"img_{i}.tif")

    config_path = _write_smoke_config(tmp_path, data_dir)

    results = main(str(config_path), backbone_loader=fake_loader)

    assert "probe_metrics" in results
    assert "lora_metrics" in results
    assert 0.0 <= results["probe_metrics"]["accuracy"] <= 1.0
    assert 0.0 <= results["lora_metrics"]["accuracy"] <= 1.0
