"""Deterministic synthetic battery aging data for Colab smoke tests.

The generated data are not a substitute for public NASA/CALCE/Oxford/Iontech
archives. They provide a lightweight, physics-shaped sanity dataset so the
single-cell Colab workflow can verify installation, plotting, SHAP plumbing, and
artifact writing before users mount large raw datasets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from battery_xai.data.schema import CANONICAL_COLUMNS


def generate_demo_raw_data(cells_per_dataset: int = 3, cycles: int = 90, points_per_cycle: int = 16, seed: int = 2026) -> pd.DataFrame:
    """Create canonical raw records with monotonic degradation trends.

    The synthetic profiles include dataset-specific fade rates, resistance
    growth, voltage windows, thermal rise, Coulombic inefficiency, SOH, and RUL
    values. This makes it useful for end-to-end CI/Colab validation while being
    clearly labeled as demo data.
    """

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | str]] = []
    dataset_rates = {"nasa": 0.0024, "calce": 0.0020, "oxford": 0.0017, "iontech": 0.0028}
    nominal_capacity = {"nasa": 2.0, "calce": 1.1, "oxford": 0.74, "iontech": 2.5}
    for dataset, fade_rate in dataset_rates.items():
        for cell_idx in range(cells_per_dataset):
            cell_id = f"demo_{dataset}_{cell_idx + 1:02d}"
            q0 = nominal_capacity[dataset] * (1.0 + rng.normal(0, 0.015))
            resistance0 = 0.035 + 0.005 * cell_idx + rng.normal(0, 0.001)
            for cycle in range(cycles):
                nonlinear = 0.000006 * cycle**1.45
                capacity = q0 * max(0.55, 1.0 - fade_rate * cycle - nonlinear + rng.normal(0, 0.0015))
                soh = capacity / q0
                resistance = resistance0 * (1.0 + 0.006 * cycle + rng.normal(0, 0.003))
                charge_capacity = capacity * (1.0 + 0.006 + rng.normal(0, 0.0008))
                coulombic = capacity / charge_capacity
                eol_candidates = np.where(q0 * (1.0 - fade_rate * np.arange(cycles) - 0.000006 * np.arange(cycles) ** 1.45) / q0 <= 0.8)[0]
                eol_cycle = int(eol_candidates[0]) if len(eol_candidates) else cycles - 1
                rul = max(eol_cycle - cycle, 0)
                for point in range(points_per_cycle):
                    frac = point / max(points_per_cycle - 1, 1)
                    voltage = 4.2 - 1.15 * frac - 0.12 * (1 - soh) + rng.normal(0, 0.004)
                    current = -1.0 - 0.05 * cell_idx + rng.normal(0, 0.01)
                    temp = 25.0 + 2.0 * abs(current) + 7.0 * resistance + 0.01 * cycle + rng.normal(0, 0.08)
                    rows.append(
                        {
                            "dataset": dataset,
                            "cell_id": cell_id,
                            "cycle_index": cycle,
                            "step_type": "discharge",
                            "time_s": float(point * 60),
                            "voltage_v": voltage,
                            "current_a": current,
                            "temperature_c": temp,
                            "capacity_ah": capacity,
                            "charge_capacity_ah": charge_capacity,
                            "discharge_capacity_ah": capacity,
                            "energy_wh": capacity * 3.65,
                            "charge_energy_wh": charge_capacity * 3.85,
                            "discharge_energy_wh": capacity * 3.65 * coulombic,
                            "internal_resistance_ohm": resistance,
                            "ambient_temperature_c": 25.0,
                            "soh": soh,
                            "rul_cycles": rul,
                        }
                    )
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
