"""Preprocessing for cycle-resolved and time-resolved battery aging records."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from battery_xai.data.schema import NUMERIC_COLUMNS


def infer_cycle_table(df: pd.DataFrame, eol_soh: float = 0.80) -> pd.DataFrame:
    """Aggregate raw records into one row per dataset/cell/cycle.

    The aggregation preserves interpretable electrochemical summaries and infers
    SOH/RUL when capacity is present.
    """

    if df.empty:
        return df.copy()
    work = df.copy().sort_values(["dataset", "cell_id", "cycle_index", "time_s"])
    grouped = work.groupby(["dataset", "cell_id", "cycle_index"], dropna=False)
    cycle = grouped.agg(
        voltage_mean_v=("voltage_v", "mean"), voltage_min_v=("voltage_v", "min"), voltage_max_v=("voltage_v", "max"),
        current_mean_a=("current_a", "mean"), current_abs_mean_a=("current_a", lambda s: float(np.nanmean(np.abs(s)))),
        temperature_mean_c=("temperature_c", "mean"), temperature_max_c=("temperature_c", "max"),
        capacity_ah=("capacity_ah", "max"), charge_capacity_ah=("charge_capacity_ah", "max"),
        discharge_capacity_ah=("discharge_capacity_ah", "max"), energy_wh=("energy_wh", "max"),
        charge_energy_wh=("charge_energy_wh", "max"), discharge_energy_wh=("discharge_energy_wh", "max"),
        internal_resistance_ohm=("internal_resistance_ohm", "mean"), ambient_temperature_c=("ambient_temperature_c", "mean"),
        observed_soh=("soh", "mean"), observed_rul_cycles=("rul_cycles", "mean"), sample_count=("time_s", "size"),
    ).reset_index()
    cap = cycle["discharge_capacity_ah"].fillna(cycle["capacity_ah"]).fillna(cycle["charge_capacity_ah"])
    cycle["capacity_ah"] = cap
    cycle["initial_capacity_ah"] = cycle.groupby(["dataset", "cell_id"])["capacity_ah"].transform(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
    inferred_soh = cycle["capacity_ah"] / cycle["initial_capacity_ah"]
    cycle["soh"] = cycle["observed_soh"].fillna(inferred_soh)
    cycle["soh"] = np.where(cycle["soh"] > 1.5, cycle["soh"] / 100.0, cycle["soh"])
    cycle["rul_cycles"] = cycle["observed_rul_cycles"]
    for _, idx in cycle.groupby(["dataset", "cell_id"]).groups.items():
        sub = cycle.loc[idx].sort_values("cycle_index")
        below = sub.loc[sub["soh"] <= eol_soh, "cycle_index"]
        eol_cycle = float(below.iloc[0]) if len(below) else float(sub["cycle_index"].max())
        cycle.loc[sub.index, "rul_cycles"] = cycle.loc[sub.index, "rul_cycles"].fillna(eol_cycle - sub["cycle_index"])
    return cycle.drop(columns=["observed_soh", "observed_rul_cycles"])


class BatteryPreprocessor:
    """Clean, align, filter, and scale cycle-level battery features."""

    def __init__(self, rolling_window: int = 5, outlier_zscore: float = 5.0, scaler: str = "standard") -> None:
        self.rolling_window = rolling_window
        self.outlier_zscore = outlier_zscore
        self.scaler_name = scaler
        self.scaler = {"standard": StandardScaler, "robust": RobustScaler, "minmax": MinMaxScaler}[scaler]()
        self.feature_columns_: list[str] = []

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values, physical outliers, smoothing, and cycle alignment."""

        work = df.copy().sort_values(["dataset", "cell_id", "cycle_index"])
        for column in work.select_dtypes(include=[np.number]).columns:
            work[column] = work.groupby(["dataset", "cell_id"])[column].transform(lambda s: s.interpolate("linear", limit_direction="both"))
            median = work[column].median()
            mad = np.nanmedian(np.abs(work[column] - median)) or 1.0
            robust_z = 0.6745 * (work[column] - median) / mad
            work.loc[robust_z.abs() > self.outlier_zscore, column] = np.nan
            work[column] = work.groupby(["dataset", "cell_id"])[column].transform(lambda s: s.interpolate("linear", limit_direction="both"))
        smooth_cols = [c for c in ["capacity_ah", "soh", "internal_resistance_ohm"] if c in work]
        for column in smooth_cols:
            work[f"{column}_smooth"] = work.groupby(["dataset", "cell_id"])[column].transform(
                lambda s: s.rolling(self.rolling_window, min_periods=1, center=True).median()
            )
        work["aligned_cycle"] = work.groupby(["dataset", "cell_id"]).cumcount()
        return work

    def fit_transform_features(self, df: pd.DataFrame, exclude: Iterable[str] = ("soh", "rul_cycles")) -> pd.DataFrame:
        """Scale numeric model features and keep targets untouched."""

        work = df.copy()
        exclude_set = set(exclude) | {"cycle_index", "aligned_cycle"}
        self.feature_columns_ = [c for c in work.select_dtypes(include=[np.number]).columns if c not in exclude_set]
        work[self.feature_columns_] = self.scaler.fit_transform(work[self.feature_columns_].fillna(0.0))
        return work

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        work[self.feature_columns_] = self.scaler.transform(work[self.feature_columns_].fillna(0.0))
        return work
