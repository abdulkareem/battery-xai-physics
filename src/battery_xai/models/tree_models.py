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
        self.feature_medians_: pd.Series | None = None

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

    def _prepare_features(self, X: pd.DataFrame | np.ndarray, fit: bool = False) -> pd.DataFrame | np.ndarray:
        """Replace non-finite feature values for estimators that cannot consume NaNs."""

        if isinstance(X, pd.DataFrame):
            frame = X.replace([np.inf, -np.inf], np.nan).copy()
            if fit or self.feature_medians_ is None:
                self.feature_medians_ = frame.median(numeric_only=True).fillna(0.0)
            return frame.fillna(self.feature_medians_)
        arr = np.asarray(X, dtype=float)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        if fit or self.feature_medians_ is None:
            self.feature_medians_ = pd.Series(np.nanmedian(arr, axis=0)).fillna(0.0)
        return np.where(np.isnan(arr), self.feature_medians_.to_numpy(), arr)

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray, **kwargs: Any) -> "SklearnTreeRegressor":
        self.model.fit(self._prepare_features(X, fit=True), y, **kwargs)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(self._prepare_features(X, fit=False)))

    @property
    def feature_importances_(self) -> np.ndarray | None:
        return getattr(self.model, "feature_importances_", None)
