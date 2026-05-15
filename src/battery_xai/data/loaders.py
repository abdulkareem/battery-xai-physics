"""Dataset ingestion for NASA, CALCE, Oxford/BIL, Iontech, and generic battery files."""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat

from battery_xai.data.schema import CANONICAL_COLUMNS, NUMERIC_COLUMNS, BatteryDatasetMetadata

LOGGER = logging.getLogger(__name__)

_COLUMN_ALIASES = {
    "cycle": "cycle_index", "cycle_number": "cycle_index", "cycle_index": "cycle_index",
    "time": "time_s", "test_time": "time_s", "time_s": "time_s",
    "voltage": "voltage_v", "voltage_v": "voltage_v", "voltage_measured": "voltage_v",
    "current": "current_a", "current_a": "current_a", "current_measured": "current_a",
    "temperature": "temperature_c", "temp": "temperature_c", "temperature_measured": "temperature_c",
    "capacity": "capacity_ah", "capacity_ah": "capacity_ah", "qdischarge": "discharge_capacity_ah",
    "charge_capacity": "charge_capacity_ah", "discharge_capacity": "discharge_capacity_ah",
    "energy": "energy_wh", "ir": "internal_resistance_ohm", "resistance": "internal_resistance_ohm",
    "soh": "soh", "rul": "rul_cycles", "step": "step_type", "type": "step_type",
    "battery_id": "cell_id", "cell": "cell_id", "cell_id": "cell_id",
}


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed: dict[str, str] = {}
    for column in df.columns:
        key = str(column).strip().lower().replace(" ", "_").replace("/", "_")
        renamed[column] = _COLUMN_ALIASES.get(key, key)
    return df.rename(columns=renamed)


def _finalize(df: pd.DataFrame, metadata: BatteryDatasetMetadata, fallback_cell: str) -> pd.DataFrame:
    df = _canonicalize_columns(df).copy()
    if "dataset" not in df:
        df["dataset"] = metadata.dataset
    if "cell_id" not in df:
        df["cell_id"] = fallback_cell
    if "cycle_index" not in df:
        df["cycle_index"] = np.arange(len(df), dtype=int)
    if "step_type" not in df:
        df["step_type"] = "cycle"
    for column in CANONICAL_COLUMNS:
        if column not in df:
            df[column] = np.nan
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["dataset"] = df["dataset"].fillna(metadata.dataset).astype(str)
    df["cell_id"] = df["cell_id"].fillna(fallback_cell).astype(str)
    df["step_type"] = df["step_type"].fillna("cycle").astype(str).str.lower()
    return df[CANONICAL_COLUMNS]


def read_tabular(path: Path) -> pd.DataFrame:
    """Read CSV, parquet, Excel, JSON, or pickle battery data."""

    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            obj = pickle.load(handle)
        if isinstance(obj, pd.DataFrame):
            return obj
        if isinstance(obj, dict):
            return pd.DataFrame(obj)
    raise ValueError(f"Unsupported tabular format: {path}")


def _flatten_mat_struct(obj: object, prefix: str = "") -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    if isinstance(obj, np.ndarray) and obj.dtype.names:
        for name in obj.dtype.names:
            arrays.update(_flatten_mat_struct(obj[name][0, 0], f"{prefix}{name}_"))
    elif isinstance(obj, np.ndarray) and obj.size > 0 and np.issubdtype(obj.dtype, np.number):
        arrays[prefix[:-1] or "value"] = np.ravel(obj)
    return arrays


def read_mat(path: Path) -> pd.DataFrame:
    """Read MATLAB battery files, including NASA-style structs and HDF5 MAT files."""

    try:
        mat = loadmat(path, squeeze_me=False, struct_as_record=True)
        arrays: dict[str, np.ndarray] = {}
        for key, value in mat.items():
            if not key.startswith("__"):
                arrays.update(_flatten_mat_struct(value, key + "_"))
        if arrays:
            min_len = min(len(v) for v in arrays.values() if len(v) > 0)
            return pd.DataFrame({k: v[:min_len] for k, v in arrays.items() if len(v) >= min_len})
    except NotImplementedError:
        pass

    rows: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as handle:
        def visitor(name: str, node: h5py.Dataset) -> None:
            if isinstance(node, h5py.Dataset) and np.issubdtype(node.dtype, np.number):
                arr = np.asarray(node).ravel()
                if arr.size:
                    rows[name.replace("/", "_")] = arr
        handle.visititems(visitor)
    if not rows:
        raise ValueError(f"No numeric arrays found in {path}")
    min_len = min(len(v) for v in rows.values())
    return pd.DataFrame({k: v[:min_len] for k, v in rows.items() if len(v) >= min_len})


class BatteryDatasetLoader:
    """Load all supported files under a dataset root into the canonical schema."""

    supported_suffixes = {".csv", ".txt", ".parquet", ".pq", ".xlsx", ".xls", ".json", ".pkl", ".pickle", ".mat"}

    def __init__(self, metadata: BatteryDatasetMetadata) -> None:
        self.metadata = metadata

    def discover_files(self) -> list[Path]:
        root = Path(self.metadata.source_path)
        if root.is_file():
            return [root]
        if not root.exists():
            LOGGER.warning("Dataset path does not exist: %s", root)
            return []
        return sorted(path for path in root.rglob("*") if path.suffix.lower() in self.supported_suffixes)

    def load_file(self, path: Path) -> pd.DataFrame:
        raw = read_mat(path) if path.suffix.lower() == ".mat" else read_tabular(path)
        return _finalize(raw, self.metadata, fallback_cell=path.stem)

    def load(self) -> pd.DataFrame:
        frames = []
        for path in self.discover_files():
            try:
                frames.append(self.load_file(path))
            except Exception as exc:  # noqa: BLE001 - continue across heterogeneous public archives
                LOGGER.exception("Failed to load %s: %s", path, exc)
        if not frames:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)
        return pd.concat(frames, ignore_index=True)


def load_datasets(dataset_configs: dict[str, dict], eol_soh: float = 0.80) -> pd.DataFrame:
    """Load enabled datasets from configuration."""

    frames = []
    for name, cfg in dataset_configs.items():
        if not cfg.get("enabled", True):
            continue
        metadata = BatteryDatasetMetadata(name, cfg["path"], cfg.get("nominal_capacity_ah"), eol_soh)
        frame = BatteryDatasetLoader(metadata).load()
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CANONICAL_COLUMNS)
