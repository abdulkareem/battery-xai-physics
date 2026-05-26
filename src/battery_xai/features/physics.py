"""Physics-guided degradation feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from sklearn.cluster import KMeans


def safe_ratio(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray) -> np.ndarray:
    denom = np.asarray(denominator, dtype=float)
    return np.divide(np.asarray(numerator, dtype=float), denom, out=np.full_like(denom, np.nan), where=np.abs(denom) > 1e-12)


def extract_dq_dv_features(voltage: np.ndarray, capacity: np.ndarray, window: int = 21, polyorder: int = 3) -> dict[str, float]:
    """Compute incremental-capacity descriptors from voltage/capacity traces."""

    mask = np.isfinite(voltage) & np.isfinite(capacity)
    voltage, capacity = voltage[mask], capacity[mask]
    if voltage.size < max(window, 8):
        return {"ica_peak_value": np.nan, "ica_peak_voltage_v": np.nan, "ica_area": np.nan, "ica_peak_count": 0.0}
    order = np.argsort(voltage)
    voltage, capacity = voltage[order], capacity[order]
    _, unique_idx = np.unique(voltage, return_index=True)
    voltage, capacity = voltage[unique_idx], capacity[unique_idx]
    if voltage.size < max(window, 8):
        return {"ica_peak_value": np.nan, "ica_peak_voltage_v": np.nan, "ica_area": np.nan, "ica_peak_count": 0.0}
    window = min(window if window % 2 else window + 1, voltage.size - (1 - voltage.size % 2))
    window = max(window, polyorder + 2 + ((polyorder + 2) % 2 == 0))
    smoothed_q = savgol_filter(capacity, window_length=window, polyorder=min(polyorder, window - 2))
    dq_dv = np.gradient(smoothed_q, voltage)
    peaks, _ = find_peaks(np.abs(dq_dv), distance=max(1, voltage.size // 20))
    peak_idx = peaks[np.nanargmax(np.abs(dq_dv[peaks]))] if len(peaks) else int(np.nanargmax(np.abs(dq_dv)))
    return {
        "ica_peak_value": float(dq_dv[peak_idx]),
        "ica_peak_voltage_v": float(voltage[peak_idx]),
        "ica_area": float(np.trapezoid(np.abs(dq_dv), voltage)),
        "ica_peak_count": float(len(peaks)),
    }


class PhysicsFeatureExtractor:
    """Create physically interpretable degradation features from cycle tables."""

    def __init__(self, early_warning_window: int = 20, cluster_count: int = 4) -> None:
        self.early_warning_window = early_warning_window
        self.cluster_count = cluster_count

    def transform(self, cycle_df: pd.DataFrame) -> pd.DataFrame:
        work = cycle_df.copy().sort_values(["dataset", "cell_id", "cycle_index"])
        keys = ["dataset", "cell_id"]
        work["capacity_fade_ah"] = work.groupby(keys)["capacity_ah"].transform(lambda s: s.iloc[0] - s)
        work["capacity_fade_rate"] = work.groupby(keys)["capacity_ah"].transform(lambda s: -s.diff().rolling(5, min_periods=1).mean())
        work["resistance_growth_ohm"] = work.groupby(keys)["internal_resistance_ohm"].transform(lambda s: s - s.iloc[0])
        work["resistance_growth_rate"] = work.groupby(keys)["internal_resistance_ohm"].transform(lambda s: s.diff().rolling(5, min_periods=1).mean())
        work["coulombic_efficiency"] = safe_ratio(work["discharge_capacity_ah"], work["charge_capacity_ah"])
        work["energy_efficiency"] = safe_ratio(work["discharge_energy_wh"], work["charge_energy_wh"])
        work["temperature_rise_c"] = work["temperature_max_c"] - work[["temperature_mean_c", "ambient_temperature_c"]].bfill(axis=1).iloc[:, 0]
        work["voltage_window_v"] = work["voltage_max_v"] - work["voltage_min_v"]
        work["charge_discharge_slope_proxy"] = safe_ratio(work["voltage_window_v"], work["capacity_ah"].abs())
        work["entropy_proxy"] = safe_ratio(work["temperature_rise_c"], work["current_abs_mean_a"] * work["voltage_window_v"])
        work["relaxation_proxy_v"] = work.groupby(keys)["voltage_mean_v"].transform(lambda s: s.diff().abs().rolling(3, min_periods=1).mean())
        work["soh_acceleration"] = work.groupby(keys)["soh"].transform(lambda s: s.diff().diff())
        work["early_warning_indicator"] = work.groupby(keys)["capacity_fade_rate"].transform(
            lambda s: s.rolling(self.early_warning_window, min_periods=3).mean()
        )
        work["physics_consistency_score"] = self.physics_consistency_score(work)
        return self._add_trajectory_clusters(work)

    def physics_consistency_score(self, df: pd.DataFrame) -> pd.Series:
        """Score consistency with monotonic capacity fade and resistance growth in [0, 1]."""

        scores = pd.Series(index=df.index, dtype=float)
        for _, sub in df.groupby(["dataset", "cell_id"]):
            cap_ok = (-sub["capacity_ah"].diff()).rolling(10, min_periods=1).mean().ge(-1e-4)
            if "internal_resistance_ohm" in sub and sub["internal_resistance_ohm"].notna().any():
                res_ok = sub["internal_resistance_ohm"].diff().rolling(10, min_periods=1).mean().ge(-1e-5)
            else:
                res_ok = pd.Series(True, index=sub.index)
            scores.loc[sub.index] = 0.5 * cap_ok.astype(float) + 0.5 * res_ok.astype(float)
        return scores.fillna(0.5)

    def _add_trajectory_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        features = ["capacity_fade_rate", "resistance_growth_rate", "temperature_rise_c", "coulombic_efficiency"]
        valid = df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if len(valid) < self.cluster_count:
            df["degradation_cluster"] = 0
            return df
        model = KMeans(n_clusters=self.cluster_count, n_init=10, random_state=2026)
        df["degradation_cluster"] = model.fit_predict(valid)
        return df
