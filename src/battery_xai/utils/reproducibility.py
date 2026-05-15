"""Reproducibility helpers for NumPy, scikit-learn, and PyTorch."""
from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def seed_everything(seed: int = 2026, deterministic: bool = True) -> None:
    """Seed all available random number generators.

    The function avoids importing PyTorch unless installed, keeping CPU-only data
    processing environments lightweight.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(False)
    except ImportError:
        return


def get_device(preference: str = "auto") -> Any:
    """Return a torch device when PyTorch is available, otherwise ``None``."""

    try:
        import torch
    except ImportError:
        return None
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)
