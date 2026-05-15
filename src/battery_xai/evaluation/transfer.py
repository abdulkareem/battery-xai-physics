"""Cross-dataset transfer-learning evaluation protocols."""
from __future__ import annotations

import pandas as pd

from battery_xai.evaluation.metrics import regression_metrics
from battery_xai.models.training import train_tree_model


def run_transfer_protocols(df: pd.DataFrame, feature_columns: list[str], target: str, protocols: list[dict], model_type: str, params: dict) -> pd.DataFrame:
    """Train on configured source datasets and evaluate on held-out target datasets."""

    rows = []
    for protocol in protocols:
        train = df[df["dataset"].isin(protocol["train"])].dropna(subset=[target])
        test = df[df["dataset"].isin(protocol["test"])].dropna(subset=[target])
        if train.empty or test.empty:
            continue
        model = train_tree_model(train, feature_columns, target, model_type, params)
        pred = model.predict(test[feature_columns])
        row = {"protocol": protocol["name"], "train_datasets": "+".join(protocol["train"]), "test_datasets": "+".join(protocol["test"])}
        row.update(regression_metrics(test[target].to_numpy(), pred, prefix=f"{target}_"))
        rows.append(row)
    return pd.DataFrame(rows)
