"""SHAP explainability including temporal drift and cross-dataset consistency."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def tree_shap(model, X: pd.DataFrame, output_dir: str | Path, max_samples: int = 2048) -> tuple[np.ndarray, pd.DataFrame]:
    """Compute and persist TreeExplainer SHAP values."""

    import shap

    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    sample = X.sample(min(max_samples, len(X)), random_state=2026) if len(X) > max_samples else X
    prepared = model._prepare_features(sample, fit=False) if hasattr(model, "_prepare_features") else sample
    explainer = shap.TreeExplainer(getattr(model, "model", model))
    values = explainer.shap_values(prepared)
    np.save(output_dir / "tree_shap_values.npy", values)
    pd.DataFrame(prepared, columns=sample.columns).to_parquet(output_dir / "tree_shap_features.parquet")
    importance = pd.DataFrame({"feature": sample.columns, "mean_abs_shap": np.abs(values).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(output_dir / "global_shap_importance.csv", index=False)
    return values, importance


def deep_shap(model, background, samples, output_dir: str | Path):
    """Compute DeepExplainer values for PyTorch temporal models."""

    import shap

    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    explainer = shap.DeepExplainer(model, background)
    values = explainer.shap_values(samples)
    np.save(output_dir / "deep_shap_values.npy", values)
    return values


def temporal_shap_evolution(shap_values: np.ndarray, metadata: pd.DataFrame, feature_names: list[str], bins: int = 8) -> pd.DataFrame:
    """Aggregate absolute SHAP values over normalized life bins."""

    values = np.asarray(shap_values)
    if values.ndim == 3:
        values = np.nanmean(np.abs(values), axis=1)
    life = metadata.groupby(["dataset", "cell_id"])["cycle_index"].transform(lambda s: (s - s.min()) / max(float(s.max() - s.min()), 1.0))
    frame = pd.DataFrame(np.abs(values), columns=feature_names).iloc[: len(metadata)].copy()
    frame["life_bin"] = pd.cut(life.iloc[: len(frame)], bins=np.linspace(0, 1, bins + 1), include_lowest=True, labels=False)
    long = frame.groupby("life_bin")[feature_names].mean().reset_index().melt("life_bin", var_name="feature", value_name="mean_abs_shap")
    return long.sort_values(["life_bin", "mean_abs_shap"], ascending=[True, False])


def shap_drift_score(temporal_importance: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
    """Quantify how feature attribution changes from early to late life."""

    pivot = temporal_importance.pivot_table(index="feature", columns="life_bin", values="mean_abs_shap", aggfunc="mean").fillna(0.0)
    if pivot.empty:
        return pd.DataFrame(columns=["feature", "drift_score"])
    first, last = pivot.columns.min(), pivot.columns.max()
    drift = (pivot[last] - pivot[first]).abs().sort_values(ascending=False).head(top_k)
    return drift.rename("drift_score").reset_index()


def cross_dataset_shap_consistency(importances: dict[str, pd.DataFrame], top_k: int = 20) -> pd.DataFrame:
    """Compute pairwise Jaccard consistency among top SHAP features across datasets."""

    rows = []
    names = sorted(importances)
    for i, left in enumerate(names):
        left_set = set(importances[left].sort_values("mean_abs_shap", ascending=False).head(top_k)["feature"])
        for right in names[i + 1 :]:
            right_set = set(importances[right].sort_values("mean_abs_shap", ascending=False).head(top_k)["feature"])
            union = left_set | right_set
            rows.append({"dataset_a": left, "dataset_b": right, "top_k_jaccard": len(left_set & right_set) / max(len(union), 1)})
    return pd.DataFrame(rows)
