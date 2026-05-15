"""Configuration loading and experiment path management."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentPaths:
    """Resolved output paths for a reproducible experiment run."""

    root: Path
    results: Path
    figures: Path
    models: Path
    shap: Path
    logs: Path

    def create(self) -> None:
        for path in (self.results, self.figures, self.models, self.shap, self.logs):
            path.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path = "configs/default.yaml", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a YAML configuration and apply shallow/dotted-key overrides.

    Overrides accept either nested dictionaries or dotted keys, e.g.
    ``{"models.epochs": 20}``.
    """

    with Path(path).open("r", encoding="utf-8") as handle:
        cfg: dict[str, Any] = yaml.safe_load(handle)
    if overrides:
        for key, value in overrides.items():
            if "." in key:
                cursor = cfg
                parts = key.split(".")
                for part in parts[:-1]:
                    cursor = cursor.setdefault(part, {})
                cursor[parts[-1]] = value
            elif isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
    return cfg


def resolve_paths(cfg: dict[str, Any], experiment_name: str | None = None) -> ExperimentPaths:
    """Create standard output folders for an experiment."""

    root = Path(".").resolve()
    suffix = experiment_name or cfg.get("project", {}).get("name", "experiment")
    paths = ExperimentPaths(
        root=root,
        results=root / cfg["project"]["output_dir"] / suffix,
        figures=root / cfg["project"]["figure_dir"] / suffix,
        models=root / cfg["project"]["model_dir"] / suffix,
        shap=root / cfg["project"]["shap_dir"] / suffix,
        logs=root / cfg["project"]["log_dir"] / suffix,
    )
    paths.create()
    return paths
