"""Command-line interface for the battery XAI pipeline."""
from __future__ import annotations

import argparse

from battery_xai.experiments.run_pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Physics-guided explainable battery degradation pipeline")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML configuration")
    parser.add_argument("--experiment", default=None, help="Experiment name used for output folders")
    args = parser.parse_args()
    outputs = run(args.config, args.experiment)
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
