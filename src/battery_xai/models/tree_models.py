"""Tree-based baselines with physics-compatible monotonic constraints where supported."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from battery_xai.models.base import BatteryRegressor


class SklearnTreeRegressor(BatteryRegressor):
    """Wrapper for Random Forest, XGBoost, and LightGBM regressors."""

    def __init__(self, model_type: str = "random_forest", params: dict[str, Any] | None = None, random_state: int = 2026) -> None:
        self.name = model_type
        self.params = params or {}
        self.random_state = random_state
        self.model = self._build_model()

    def _build_model(self):
        if self.name == "random_forest":
            return RandomForestRegressor(random_state=self.random_state, n_jobs=-1, **self.params)
        if self.name == "xgboost":
            from xgboost import XGBRegressor

            return XGBRegressor(random_state=self.random_state, n_jobs=-1, tree_method="hist", **self.params)
        if self.name == "lightgbm":
            from lightgbm import LGBMRegressor

            return LGBMRegressor(random_state=self.random_state, n_jobs=-1, verbose=-1, **self.params)
        raise ValueError(f"Unknown tree model: {self.name}")

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray, **kwargs: Any) -> "SklearnTreeRegressor":
        self.model.fit(X, y, **kwargs)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(X))

    @property
    def feature_importances_(self) -> np.ndarray | None:
        return getattr(self.model, "feature_importances_", None)
