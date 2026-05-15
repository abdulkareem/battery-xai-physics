import numpy as np
import pandas as pd

from battery_xai.data.preprocessing import infer_cycle_table


def test_infer_cycle_table_computes_soh_and_rul():
    df = pd.DataFrame({
        "dataset": ["d"] * 6,
        "cell_id": ["c"] * 6,
        "cycle_index": [0, 0, 1, 1, 2, 2],
        "step_type": ["discharge"] * 6,
        "time_s": [0, 1, 0, 1, 0, 1],
        "voltage_v": [4, 3, 4, 3, 4, 3],
        "current_a": [-1] * 6,
        "temperature_c": [25] * 6,
        "capacity_ah": [2.0, 2.0, 1.8, 1.8, 1.5, 1.5],
        "charge_capacity_ah": [np.nan] * 6,
        "discharge_capacity_ah": [2.0, 2.0, 1.8, 1.8, 1.5, 1.5],
        "energy_wh": [np.nan] * 6,
        "charge_energy_wh": [np.nan] * 6,
        "discharge_energy_wh": [np.nan] * 6,
        "internal_resistance_ohm": [0.05] * 6,
        "ambient_temperature_c": [25] * 6,
        "soh": [np.nan] * 6,
        "rul_cycles": [np.nan] * 6,
    })
    cycle = infer_cycle_table(df, eol_soh=0.8)
    assert list(cycle["soh"].round(2)) == [1.0, 0.9, 0.75]
    assert cycle.loc[0, "rul_cycles"] == 2
