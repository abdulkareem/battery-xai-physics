"""Training orchestration for classical and neural battery models."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from battery_xai.evaluation.metrics import regression_metrics
from battery_xai.models.losses import PhysicsAwareLoss
from battery_xai.models.torch_models import SequenceDataset, TransformerRegressor, _LSTMModule, save_torch_checkpoint
from battery_xai.models.tree_models import SklearnTreeRegressor


def split_by_cell(df: pd.DataFrame, test_size: float = 0.2, seed: int = 2026) -> tuple[np.ndarray, np.ndarray]:
    groups = df["dataset"].astype(str) + "::" + df["cell_id"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(splitter.split(df, groups=groups))


def train_tree_model(train_df: pd.DataFrame, feature_columns: list[str], target: str, model_type: str, params: dict[str, Any]) -> SklearnTreeRegressor:
    model = SklearnTreeRegressor(model_type=model_type, params=params)
    model.fit(train_df[feature_columns], train_df[target].to_numpy())
    return model


def train_torch_temporal(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    model_kind: str = "lstm",
    output_dir: str | Path = "models/temporal",
    sequence_length: int = 32,
    batch_size: int = 128,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    hidden_size: int = 128,
    dropout: float = 0.15,
    patience: int = 15,
    device: str = "cuda",
    mixed_precision: bool = True,
) -> Any:
    """Train LSTM or Transformer with checkpointing, AMP, and early stopping."""

    import torch
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter

    device_obj = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    input_dim = len(feature_columns)
    model = _LSTMModule(input_dim, hidden_size, 2, dropout) if model_kind == "lstm" else TransformerRegressor(input_dim, hidden_size, dropout=dropout)
    model.to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = PhysicsAwareLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision and device_obj.type == "cuda")
    output_dir = Path(output_dir)
    writer = SummaryWriter(output_dir / "tensorboard")
    log_path = output_dir / "training_log.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = SequenceDataset(train_df[feature_columns].to_numpy(float), train_df[target].to_numpy(float), sequence_length).dataset
    val_ds = SequenceDataset(val_df[feature_columns].to_numpy(float), val_df[target].to_numpy(float), sequence_length).dataset
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    best_val, bad_epochs = np.inf, 0
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss"]).writeheader()
    for epoch in range(1, epochs + 1):
        model.train(); train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device_obj), yb.to(device_obj)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=mixed_precision and device_obj.type == "cuda"):
                pred = model(xb)
                loss = loss_fn(pred, yb)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            train_losses.append(float(loss.detach().cpu()))
        model.eval(); val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device_obj), yb.to(device_obj)
                val_losses.append(float(loss_fn(model(xb), yb).detach().cpu()))
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": float(np.mean(val_losses)) if val_losses else np.nan}
        writer.add_scalar("loss/train", row["train_loss"], epoch); writer.add_scalar("loss/val", row["val_loss"], epoch)
        with log_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=row.keys()).writerow(row)
        if row["val_loss"] < best_val:
            best_val, bad_epochs = row["val_loss"], 0
            save_torch_checkpoint(model, optimizer, epoch, output_dir / "best.pt", {"val_loss": best_val})
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break
    writer.close()
    return model


def evaluate_model(model: Any, df: pd.DataFrame, feature_columns: list[str], target: str) -> dict[str, float]:
    pred = model.predict(df[feature_columns])
    return regression_metrics(df[target].to_numpy(), pred, prefix=f"{target}_")
