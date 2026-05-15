"""Benchmark, ablation, and statistical tables for manuscripts."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy import stats


def save_latex_table(df: pd.DataFrame, path: str | Path, caption: str, label: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    latex = df.to_latex(index=False, float_format="%.4f", caption=caption, label=label, escape=False)
    Path(path).write_text(latex, encoding="utf-8")


def paired_significance_table(results: pd.DataFrame, metric: str, group_col: str = "model", dataset_col: str = "protocol") -> pd.DataFrame:
    """Pairwise Wilcoxon tests across matched protocols/datasets."""

    rows = []
    pivot = results.pivot_table(index=dataset_col, columns=group_col, values=metric)
    models = list(pivot.columns)
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            paired = pivot[[a, b]].dropna()
            if len(paired) < 2:
                p_value = float("nan")
            else:
                p_value = float(stats.wilcoxon(paired[a], paired[b]).pvalue)
            rows.append({"model_a": a, "model_b": b, "metric": metric, "p_value": p_value})
    return pd.DataFrame(rows)


def ablation_table(rows: list[dict], output_csv: str | Path, output_tex: str | Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    save_latex_table(df, output_tex, "Ablation study of physics-guided XAI components.", "tab:ablation")
    return df
