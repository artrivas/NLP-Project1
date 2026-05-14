"""Utility helpers for reproducible experiments."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set common random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml_config(path: str) -> dict[str, Any]:
    """Load a YAML configuration file and return it as a dictionary."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"YAML config not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML config at {config_path}: {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError(f"YAML config must contain a mapping at top level: {config_path}")

    return config


def get_device() -> torch.device:
    """Return the best available PyTorch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_cache_dir() -> str | None:
    """Return an optional Hugging Face cache directory from the environment."""
    return os.getenv("HF_HOME") or os.getenv("HF_DATASETS_CACHE")
