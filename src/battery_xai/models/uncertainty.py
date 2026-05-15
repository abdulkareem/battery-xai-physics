"""Uncertainty estimation and calibration."""
from __future__ import annotations

import numpy as np


def mc_dropout_predict(model, dataloader, samples: int = 50, device: str = "cuda") -> dict[str, np.ndarray]:
    """Estimate predictive uncertainty with Monte Carlo dropout."""

    import torch

    device_obj = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model.to(device_obj)
    model.train()
    predictions = []
    with torch.no_grad():
        for _ in range(samples):
            batch_preds = []
            for xb, _ in dataloader:
                batch_preds.append(model(xb.to(device_obj)).detach().cpu().numpy().ravel())
            predictions.append(np.concatenate(batch_preds))
    arr = np.vstack(predictions)
    return {"mean": arr.mean(axis=0), "std": arr.std(axis=0), "lower": np.quantile(arr, 0.05, axis=0), "upper": np.quantile(arr, 0.95, axis=0)}


def conformal_interval(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.10) -> float:
    """Return split-conformal absolute residual quantile."""

    residuals = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    n = residuals.size
    rank = int(np.ceil((n + 1) * (1 - alpha))) - 1
    return float(np.sort(residuals)[min(max(rank, 0), n - 1)])


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    return float(np.mean((y_true >= lower) & (y_true <= upper)))
