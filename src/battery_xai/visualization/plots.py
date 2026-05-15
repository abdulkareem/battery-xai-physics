"""Publication-quality plotting utilities."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def set_publication_style(style: str = "seaborn-v0_8-whitegrid") -> None:
    plt.style.use(style)
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 600, "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12, "legend.frameon": False})


def _save(fig, output: str | Path, formats: tuple[str, ...] = ("png", "pdf")) -> None:
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output.with_suffix(f".{fmt}"), bbox_inches="tight")
    plt.close(fig)


def plot_capacity_fade(df: pd.DataFrame, output: str | Path) -> None:
    set_publication_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    sns.lineplot(data=df, x="cycle_index", y="soh", hue="dataset", units="cell_id", estimator=None, alpha=0.35, ax=ax)
    ax.axhline(0.8, color="black", linestyle="--", linewidth=1, label="80% EOL")
    ax.set(xlabel="Cycle number", ylabel="State of health", title="Capacity fade trajectories")
    _save(fig, output)


def plot_shap_summary_bar(importance: pd.DataFrame, output: str | Path, top_n: int = 20) -> None:
    set_publication_style()
    top = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, max(3.5, 0.22 * len(top))))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#2a6fbb")
    ax.set(xlabel="Mean |SHAP|", ylabel="", title="Global physics-feature importance")
    _save(fig, output)


def plot_temporal_shap(temporal: pd.DataFrame, output: str | Path, top_n: int = 10) -> None:
    set_publication_style()
    top_features = temporal.groupby("feature")["mean_abs_shap"].mean().nlargest(top_n).index
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    sns.lineplot(data=temporal[temporal["feature"].isin(top_features)], x="life_bin", y="mean_abs_shap", hue="feature", marker="o", ax=ax)
    ax.set(xlabel="Normalized life bin", ylabel="Mean |SHAP|", title="Temporal evolution of degradation explanations")
    _save(fig, output)


def plot_transfer_results(results: pd.DataFrame, output: str | Path, metric: str = "soh_rmse") -> None:
    set_publication_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    sns.barplot(data=results, x="protocol", y=metric, ax=ax, color="#4c956c")
    ax.tick_params(axis="x", rotation=25)
    ax.set(xlabel="Transfer protocol", ylabel=metric.upper(), title="Cross-dataset generalization")
    _save(fig, output)


def plot_uncertainty(cycles: np.ndarray, y_true: np.ndarray, mean: np.ndarray, lower: np.ndarray, upper: np.ndarray, output: str | Path) -> None:
    set_publication_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(cycles, y_true, label="Observed", color="black", linewidth=1.5)
    ax.plot(cycles, mean, label="Predicted", color="#2a6fbb")
    ax.fill_between(cycles, lower, upper, color="#2a6fbb", alpha=0.2, label="90% interval")
    ax.set(xlabel="Cycle", ylabel="SOH/RUL", title="Predictive uncertainty")
    ax.legend()
    _save(fig, output)


def plot_attention_map(attention: np.ndarray, output: str | Path) -> None:
    set_publication_style()
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    sns.heatmap(attention, cmap="viridis", ax=ax)
    ax.set(xlabel="Key cycle", ylabel="Query cycle", title="Transformer attention map")
    _save(fig, output)
