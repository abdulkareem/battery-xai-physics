# Physics-Guided Explainable AI for Cross-Dataset Lithium-Ion Battery Degradation Analysis Using SHAP

This repository is a publication-level research pipeline for SOH and RUL prediction, cross-dataset generalization, physics-guided feature engineering, SHAP explainability, temporal degradation interpretation, uncertainty estimation, and manuscript artifact generation.

## Repository phases and files

### Phase 1 — Repository structure
- `configs/default.yaml`: full experiment configuration for data paths, features, models, transfer protocols, SHAP, uncertainty, and figures.
- `requirements.txt`, `environment.yml`, `setup.py`: reproducible pip/conda/GitHub installation.
- `notebooks/colab_reproducible_pipeline.ipynb`: Colab Pro+ entry point with editable install and pipeline execution.

### Phase 2 — Dataset loaders
- `src/battery_xai/data/schema.py`: canonical heterogeneous battery schema.
- `src/battery_xai/data/loaders.py`: CSV, MAT, parquet, pickle, Excel, JSON ingestion for NASA, CALCE, Oxford/BIL, and Iontech-style archives.
- `src/battery_xai/data/preprocessing.py`: missing-value interpolation, robust outlier handling, smoothing, cycle aggregation, alignment, and scaling.

### Phase 3 — Feature engineering
- `src/battery_xai/features/physics.py`: capacity fade, resistance growth, Coulombic efficiency, energy efficiency, dQ/dV descriptors, temperature rise, relaxation proxies, slopes, entropy proxy, early warnings, degradation clusters, and physics-consistency score.

### Phase 4 — Baseline and temporal models
- `src/battery_xai/models/tree_models.py`: Random Forest, XGBoost, and LightGBM wrappers.
- `src/battery_xai/models/torch_models.py`: LSTM and Transformer temporal regressors with checkpoint utilities.
- `src/battery_xai/models/training.py`: cell-wise splits, tree training, neural training with AMP, TensorBoard, CSV logging, checkpointing, and early stopping.

### Phase 5 — Explainability
- `src/battery_xai/explain/shap_analysis.py`: SHAP TreeExplainer, DeepExplainer, temporal SHAP evolution, SHAP drift, and cross-dataset consistency.

### Phase 6 — Physics-informed learning
- `src/battery_xai/models/losses.py`: monotonic SOH and temporal consistency losses.
- `src/battery_xai/models/uncertainty.py`: Monte Carlo dropout, conformal intervals, and interval coverage.

### Phase 7 — Visualization
- `src/battery_xai/visualization/plots.py`: high-resolution capacity fade curves, SHAP summaries, temporal SHAP, transfer comparisons, uncertainty plots, and attention maps.

### Phase 8 — Paper-generation utilities
- `src/battery_xai/paper/generate.py`: abstract, methodology, experimental setup, result analysis, and conclusion draft.
- `src/battery_xai/paper/tables.py`: LaTeX benchmark/ablation tables and statistical significance tests.


## Single-cell Google Colab run

Use `notebooks/single_cell_colab_pipeline.ipynb` for the requested one-cell Colab workflow. The notebook contains exactly one executable cell that:

1. Optionally mounts Google Drive.
2. Clones this repository when `REPO_URL` is provided, or uses an already-uploaded `PROJECT_DIR`.
3. Installs the package with the full scientific stack via `pip install -e .[full]`.
4. Runs `configs/colab_single_cell.yaml`.
5. Enables deterministic demo data when public datasets are not yet mounted, so the full pipeline still generates artifacts for a smoke test.
6. Displays the transfer table and a publication-style capacity-fade figure directly in Colab.

Minimal Colab usage:

```python
REPO_URL = "https://github.com/<your-user>/battery-xai-physics.git"
# Run the single cell in notebooks/single_cell_colab_pipeline.ipynb
```

For real experiments, upload or mount public archives under `data/raw/nasa`, `data/raw/calce`, `data/raw/oxford`, and `data/raw/iontech`, then set `USE_DEMO_DATA_IF_MISSING = False` at the top of the same cell.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[full]
battery-xai --config configs/default.yaml --experiment first_run
```

Place datasets under:

```text
data/raw/nasa/
data/raw/calce/
data/raw/oxford/
data/raw/iontech/
```

The loader accepts `.csv`, `.txt`, `.parquet`, `.pq`, `.xlsx`, `.xls`, `.json`, `.pkl`, `.pickle`, and `.mat` files. Column aliases are normalized to the canonical schema. For archives with nested MATLAB structs, the loader recursively extracts numeric arrays and logs unsupported files without stopping the experiment.

## Outputs

A completed run writes:

- `results/<experiment>/canonical_raw.parquet`
- `results/<experiment>/physics_features.parquet`
- `results/<experiment>/transfer_results.csv`
- `results/<experiment>/transfer_results.tex`
- `results/<experiment>/paper_draft.md`
- `models/<experiment>/*.joblib` and neural checkpoints
- `figures/<experiment>/*.png` and `*.pdf`
- `shap_outputs/<experiment>/*`
- `logs/<experiment>/battery_xai.log`

## Cross-dataset protocols

Default transfer protocols are configured as:

1. NASA → CALCE
2. CALCE → Oxford/Battery Intelligence Lab
3. NASA + CALCE + Oxford → Iontech

Add additional protocols in `configs/default.yaml` without changing source code.

## Colab Pro+ notes

The runner automatically attempts Google Drive mounting when running inside Colab, supports CUDA devices including A100/H100/T4, uses mixed precision for neural models, stores checkpoints, and can resume from saved artifacts by reusing output directories.

## Testing

```bash
pytest -q
```
