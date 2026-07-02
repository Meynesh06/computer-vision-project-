from tissue_classifier.config import load_config


def test_local_smoke_config_is_valid():
    config = load_config("configs/local_smoke.yaml")
    assert config["device"] == "auto"
    assert config["precision"] == "fp32"


def test_linear_probe_config_is_valid():
    config = load_config("configs/linear_probe.yaml")
    assert config["backbone"] == "hf-hub:paige-ai/Virchow"


def test_lora_finetune_config_is_valid():
    config = load_config("configs/lora_finetune.yaml")
    assert "lora_r" in config
    assert "lora_alpha" in config
    assert "target_modules" in config
