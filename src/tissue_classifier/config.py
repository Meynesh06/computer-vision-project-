from pathlib import Path

import torch
import yaml

REQUIRED_KEYS = ["data_dir", "device", "precision", "backbone", "batch_size", "seed"]


class ConfigError(ValueError):
    pass


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)

    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ConfigError(f"Config {path} is missing required keys: {missing}")

    return config


def resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        raise ConfigError("device 'cuda' was requested but CUDA is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ConfigError("device 'mps' was requested but MPS is not available")

    return requested


def resolve_precision(requested: str, device: str) -> str:
    if requested != "auto":
        return requested

    if device != "cuda":
        return "fp32"

    major, _minor = torch.cuda.get_device_capability(device=0)
    return "bf16" if major >= 8 else "fp16"
