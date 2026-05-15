"""Evaluation metrics for SOH and RUL prediction."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> dict[str, float]:
    """Return RMSE, MAE, R2, MAPE, and bias."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if y_true.size == 0:
        return {f"{prefix}{k}": np.nan for k in ["rmse", "mae", "r2", "mape", "bias"]}
    denom = np.where(np.abs(y_true) < 1e-12, np.nan, np.abs(y_true))
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if y_true.size > 1 else np.nan,
        "mape": float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0),
        "bias": float(np.mean(y_pred - y_true)),
    }
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def soh_error(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return regression_metrics(y_true, y_pred, prefix="soh_")


def rul_error(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return regression_metrics(y_true, y_pred, prefix="rul_")
