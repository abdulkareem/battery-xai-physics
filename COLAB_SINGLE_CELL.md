# One-cell Google Colab runner

Open `notebooks/single_cell_colab_pipeline.ipynb` in Google Colab. It contains exactly one executable cell. Set `REPO_URL` to your GitHub repository URL if Colab has not already cloned/uploaded the repo, then run the cell.

The single cell installs the package, optionally mounts Drive, runs `configs/colab_single_cell.yaml`, falls back to deterministic demo data only when real public archives are missing, and displays generated outputs.

For production results, place datasets under:

```text
data/raw/nasa/
data/raw/calce/
data/raw/oxford/
data/raw/iontech/
```

Then set `USE_DEMO_DATA_IF_MISSING = False` in the cell before running.
