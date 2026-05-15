"""PyTorch temporal models for battery degradation."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import numpy as np


class SequenceDataset:
    """Windowed cycle sequences for SOH/RUL forecasting."""

    def __init__(self, X: np.ndarray, y: np.ndarray, sequence_length: int = 32) -> None:
        import torch
        from torch.utils.data import Dataset

        class _Dataset(Dataset):
            def __init__(self, features: np.ndarray, targets: np.ndarray, seq_len: int) -> None:
                self.X = torch.as_tensor(features, dtype=torch.float32)
                self.y = torch.as_tensor(targets, dtype=torch.float32).view(-1, 1)
                self.seq_len = seq_len

            def __len__(self) -> int:
                return max(0, len(self.X) - self.seq_len + 1)

            def __getitem__(self, idx: int):
                end = idx + self.seq_len
                return self.X[idx:end], self.y[end - 1]

        self.dataset = _Dataset(X, y, sequence_length)


class LSTMRegressor:
    """Multi-layer LSTM regressor with dropout for MC uncertainty."""

    def __init__(self, input_dim: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.15) -> None:
        import torch

        self.module = torch.nn.Sequential()
        self.model = _LSTMModule(input_dim, hidden_size, num_layers, dropout)

    def __getattr__(self, name):
        if name == "model":
            raise AttributeError
        return getattr(self.model, name)


class _LSTMModule(__import__("torch").nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        import torch

        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Sequential(torch.nn.LayerNorm(hidden_size), torch.nn.Linear(hidden_size, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.dropout(out[:, -1, :]))


class PositionalEncoding(__import__("torch").nn.Module):
    """Sinusoidal positional encoding for cycle sequences."""

    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        import torch

        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerRegressor(__import__("torch").nn.Module):
    """Transformer encoder for temporal battery aging signatures."""

    def __init__(self, input_dim: int, hidden_size: int = 128, heads: int = 4, layers: int = 3, dropout: float = 0.15) -> None:
        import torch

        super().__init__()
        self.input_projection = torch.nn.Linear(input_dim, hidden_size)
        self.position = PositionalEncoding(hidden_size)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=heads, dim_feedforward=hidden_size * 4, dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Sequential(torch.nn.LayerNorm(hidden_size), torch.nn.Linear(hidden_size, 1))
        self.last_attention_map = None

    def forward(self, x):
        z = self.position(self.input_projection(x))
        encoded = self.encoder(z)
        return self.head(self.dropout(encoded[:, -1, :]))


def save_torch_checkpoint(model, optimizer, epoch: int, path: str | Path, metrics: dict | None = None) -> None:
    import torch

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "metrics": metrics or {}}, path)
