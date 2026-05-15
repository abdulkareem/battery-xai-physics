"""Canonical schema for heterogeneous lithium-ion aging datasets."""
from __future__ import annotations

from dataclasses import dataclass

CANONICAL_COLUMNS = [
    "dataset", "cell_id", "cycle_index", "step_type", "time_s", "voltage_v", "current_a",
    "temperature_c", "capacity_ah", "charge_capacity_ah", "discharge_capacity_ah",
    "energy_wh", "charge_energy_wh", "discharge_energy_wh", "internal_resistance_ohm",
    "ambient_temperature_c", "soh", "rul_cycles",
]

NUMERIC_COLUMNS = [column for column in CANONICAL_COLUMNS if column not in {"dataset", "cell_id", "step_type"}]


@dataclass(frozen=True)
class BatteryDatasetMetadata:
    """Metadata attached to all loaded records from a dataset."""

    dataset: str
    source_path: str
    nominal_capacity_ah: float | None = None
    eol_soh: float = 0.80
