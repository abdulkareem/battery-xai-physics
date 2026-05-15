import numpy as np
import pandas as pd

from battery_xai.features.physics import PhysicsFeatureExtractor, extract_dq_dv_features


def synthetic_cycle_df(n=40):
    return pd.DataFrame({
        "dataset": ["synthetic"] * n,
        "cell_id": ["cell_1"] * n,
        "cycle_index": np.arange(n),
        "capacity_ah": 2.0 - 0.01 * np.arange(n),
        "charge_capacity_ah": 2.05 - 0.008 * np.arange(n),
        "discharge_capacity_ah": 2.0 - 0.01 * np.arange(n),
        "charge_energy_wh": 7.6 - 0.02 * np.arange(n),
        "discharge_energy_wh": 7.2 - 0.025 * np.arange(n),
        "internal_resistance_ohm": 0.04 + 0.0005 * np.arange(n),
        "temperature_mean_c": 25 + 0.01 * np.arange(n),
        "temperature_max_c": 27 + 0.02 * np.arange(n),
        "ambient_temperature_c": 25,
        "voltage_mean_v": 3.7,
        "voltage_min_v": 3.0,
        "voltage_max_v": 4.2,
        "current_abs_mean_a": 1.0,
        "soh": (2.0 - 0.01 * np.arange(n)) / 2.0,
        "rul_cycles": n - np.arange(n),
    })


def test_physics_features_are_created():
    out = PhysicsFeatureExtractor(cluster_count=3).transform(synthetic_cycle_df())
    for column in ["capacity_fade_ah", "resistance_growth_ohm", "coulombic_efficiency", "physics_consistency_score", "degradation_cluster"]:
        assert column in out.columns
    assert out["physics_consistency_score"].between(0, 1).all()


def test_dq_dv_features():
    voltage = np.linspace(3.0, 4.2, 100)
    capacity = 2.0 / (1 + np.exp(-(voltage - 3.6) * 8))
    features = extract_dq_dv_features(voltage, capacity, window=11)
    assert features["ica_peak_count"] >= 0
    assert np.isfinite(features["ica_area"])
