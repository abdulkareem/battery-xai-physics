"""End-to-end experiment runner."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from battery_xai.config import load_config, resolve_paths
from battery_xai.data.demo import generate_demo_raw_data
from battery_xai.data.loaders import load_datasets
from battery_xai.data.preprocessing import BatteryPreprocessor, infer_cycle_table
from battery_xai.evaluation.transfer import run_transfer_protocols
from battery_xai.explain.shap_analysis import temporal_shap_evolution, tree_shap
from battery_xai.features.physics import PhysicsFeatureExtractor
from battery_xai.models.training import split_by_cell, train_tree_model
from battery_xai.paper.generate import write_paper_draft
from battery_xai.paper.tables import save_latex_table
from battery_xai.utils.colab import maybe_mount_drive
from battery_xai.utils.logging import configure_logging
from battery_xai.utils.reproducibility import seed_everything
from battery_xai.visualization.plots import plot_capacity_fade, plot_shap_summary_bar, plot_temporal_shap, plot_transfer_results


def run(config_path: str = "configs/default.yaml", experiment_name: str | None = None) -> dict[str, Path]:
    """Run the complete publication pipeline on available data."""

    cfg = load_config(config_path)
    seed_everything(cfg["project"]["seed"], cfg["project"].get("deterministic", True))
    maybe_mount_drive(cfg["project"].get("colab", {}).get("mount_drive", False))
    paths = resolve_paths(cfg, experiment_name)
    logger = configure_logging(paths.logs)
    logger.info("Loading datasets")
    raw = load_datasets(cfg["data"]["datasets"], cfg["data"].get("eol_soh", 0.80))
    if raw.empty and cfg["data"].get("demo_if_missing", False):
        logger.warning("No raw datasets found; generating deterministic demo data for a Colab smoke run.")
        raw = generate_demo_raw_data(
            cells_per_dataset=cfg["data"].get("demo_cells_per_dataset", 3),
            cycles=cfg["data"].get("demo_cycles", 90),
            points_per_cycle=cfg["data"].get("demo_points_per_cycle", 16),
            seed=cfg["project"]["seed"],
        )
    raw.to_parquet(paths.results / "canonical_raw.parquet")
    if raw.empty:
        logger.warning("No data found. Add public datasets under data/raw/* or set data.demo_if_missing=true.")
        write_paper_draft(paths.results / "paper_draft.md")
        return {"results": paths.results}
    cycle = infer_cycle_table(raw, cfg["data"].get("eol_soh", 0.80))
    cycle = BatteryPreprocessor(cfg["preprocessing"]["rolling_window"], cfg["preprocessing"]["outlier_zscore"]).clean(cycle)
    featured = PhysicsFeatureExtractor(cfg["features"]["early_warning_window"], cfg["features"]["trajectory_cluster_count"]).transform(cycle)
    featured.to_parquet(paths.results / "physics_features.parquet")
    target = cfg["models"]["target"]
    feature_columns = [c for c in featured.select_dtypes("number").columns if c not in {"soh", "rul_cycles"}]
    train_idx, test_idx = split_by_cell(featured.dropna(subset=[target]), seed=cfg["project"]["seed"])
    train_df = featured.dropna(subset=[target]).iloc[train_idx]
    test_df = featured.dropna(subset=[target]).iloc[test_idx]
    model_cfg = cfg["models"]["tree"]["xgboost"]
    try:
        model = train_tree_model(train_df, feature_columns, target, "xgboost", model_cfg)
    except ImportError:
        logger.warning("XGBoost unavailable; falling back to Random Forest")
        model_cfg = cfg["models"]["tree"]["random_forest"]
        model = train_tree_model(train_df, feature_columns, target, "random_forest", model_cfg)
    model.save(paths.models / f"{target}_{model.name}.joblib")
    shap_values, importance = tree_shap(model, test_df[feature_columns], paths.shap, cfg["explainability"]["explanation_samples"])
    temporal = temporal_shap_evolution(shap_values, test_df.reset_index(drop=True), feature_columns, cfg["explainability"]["temporal_bins"])
    temporal.to_csv(paths.shap / "temporal_shap_evolution.csv", index=False)
    transfer = run_transfer_protocols(featured, feature_columns, target, cfg["transfer"]["protocols"], model.name, model_cfg)
    transfer.to_csv(paths.results / "transfer_results.csv", index=False)
    save_latex_table(transfer, paths.results / "transfer_results.tex", "Cross-dataset transfer performance.", "tab:transfer")
    plot_capacity_fade(featured, paths.figures / "capacity_fade")
    plot_shap_summary_bar(importance, paths.figures / "shap_summary")
    plot_temporal_shap(temporal, paths.figures / "temporal_shap")
    if not transfer.empty:
        plot_transfer_results(transfer, paths.figures / "transfer_results", metric=f"{target}_rmse")
    write_paper_draft(paths.results / "paper_draft.md")
    return {"results": paths.results, "figures": paths.figures, "models": paths.models, "shap": paths.shap}
