"""Unified model interfaces."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


class BatteryRegressor(ABC):
    """Common estimator API for SOH and RUL models."""

    name: str

    @abstractmethod
    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray, **kwargs: Any) -> "BatteryRegressor":
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "BatteryRegressor":
        return joblib.load(path)
