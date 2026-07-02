import pytest
from tissue_classifier.config import ConfigError, load_config, resolve_device, resolve_precision


REQUIRED_KEYS = """
data_dir: /tmp/data
device: auto
precision: auto
backbone: hf-hub:paige-ai/Virchow
batch_size: 32
seed: 0
"""


def test_load_config_reads_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(REQUIRED_KEYS)

    config = load_config(config_path)

    assert config["data_dir"] == "/tmp/data"
    assert config["batch_size"] == 32


def test_load_config_raises_on_missing_key(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("data_dir: /tmp/data\n")

    with pytest.raises(ConfigError, match="device"):
        load_config(config_path)


def test_resolve_device_auto_prefers_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    assert resolve_device("auto") == "cuda"


def test_resolve_device_auto_falls_back_to_mps(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    assert resolve_device("auto") == "mps"


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    assert resolve_device("auto") == "cpu"


def test_resolve_device_explicit_cuda_raises_if_unavailable(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(ConfigError, match="cuda"):
        resolve_device("cuda")


def test_resolve_precision_auto_on_cpu_is_fp32():
    assert resolve_precision("auto", "cpu") == "fp32"


def test_resolve_precision_auto_on_cuda_prefers_bf16(monkeypatch):
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda device=0: (8, 0))
    assert resolve_precision("auto", "cuda") == "bf16"


def test_resolve_precision_auto_on_old_cuda_is_fp16(monkeypatch):
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda device=0: (7, 5))
    assert resolve_precision("auto", "cuda") == "fp16"
