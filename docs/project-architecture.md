# Project Architecture

This repository is organized as a research and ML pipeline for dendrite and spine analysis.
New files should not be created in the repository root unless they are top-level project files.

## Root Directory

The repository root is reserved for project-level files only:

- `README.md`
- `AGENTS.md`
- `.gitignore`
- `pyproject.toml`
- lock files or environment files, if added later
- `CGAL/`, while the current setup expects CGAL bindings at the root

Do not place notebooks, experiment outputs, datasets, ad-hoc scripts, plots, CSV files, JSON files, or model artifacts in the root.

## Source Code

Library code lives under `src/`.

```text
src/
  dendrite_analysis/          # dendrite domain library
  spine_analysis/             # spine domain library
  confocal_surface_repair/    # confocal repair tools
  spinetool/                  # project-level utilities
```

Use `src/spinetool/` for shared project infrastructure:

- path helpers
- config loading
- common IO helpers
- notebook bootstrap code
- pipeline utilities that do not belong to only one domain package

Use the domain packages for domain logic:

- `src/dendrite_analysis/` for dendrite, branch, neuron, network, and dendrite metric code
- `src/spine_analysis/` for spine meshes, spine metrics, clustering, classification, and segmentation code
- `src/confocal_surface_repair/` for confocal surface repair logic and resources

Do not add new Python modules to the repository root. If a script is meant to be run directly, put it in `scripts/`. If it is reusable code, put it in `src/`.

## Notebooks

All notebooks live under `notebooks/`.

```text
notebooks/
  dendrite/        # dendrite analysis, classification, comparison, segmentation
  spine/           # spine classification, clustering, manual labeling
  neuron/          # neuron-level and network-level analysis
  metrics/         # metric computation and validation notebooks
  modeling/        # model training and feature experiments
  preprocessing/   # dataset preparation and preprocessing exploration
  repair/          # surface/confocal repair notebooks
  exploratory/     # temporary exploration and scratch notebooks
  legacy/          # old notebooks kept for reference
```

When creating a new notebook, choose the narrowest matching folder. Use `notebooks/exploratory/` only for temporary work.

Notebook code should use shared path helpers from `src/spinetool/` once they are available. Until then, notebooks should resolve the project root explicitly before reading or writing project files.

## Data

Data lives under `data/`.

```text
data/
  raw/             # immutable source data
  external/        # third-party or exported external datasets
  examples/        # small example datasets used in docs or demos
  interim/         # intermediate files that are not canonical outputs
  processed/       # canonical processed dataset used for analysis and ML
```

The processed dataset layout is:

```text
data/processed/
  preprocessing/
    merged_branch_index.parquet
    preprocessing_qc.parquet
    shaft_meshes/
      <branch_id>.off
    merged_skeletons/
      <branch_id>.npy
  metadata/
    branches.parquet
    spines.parquet
  metrics/
    branch_T_metrics.parquet
    branch_P_metrics.parquet
    branch_PS_metrics.parquet
    spine_S_metrics.parquet
    mesh_qc_metrics.parquet
    branch_radius_profiles.parquet
    branch_spatial_curves.parquet
  model_inputs/
    T_geometry.zarr
    S_pointclouds.zarr
    object_index.parquet
  analysis_results/
    group_statistics.parquet
    autocorrelation_pvalues.parquet
    domain_shift_statistics.parquet
  config/
    metrics_config.yaml
    metrics_schema.json
```

Use `data/raw/` for original files that should not be modified.
Use `data/processed/` for stable artifacts that downstream analysis and models depend on.
Use `data/interim/` for temporary pipeline products that can be regenerated.

## Scripts

Runnable project scripts live under `scripts/`.

Examples:

- `scripts/preprocess_dataset.py`
- `scripts/compute_metrics.py`
- `scripts/train_model.py`
- `scripts/evaluate_model.py`

Scripts should be thin entry points. Reusable logic belongs in `src/`.

## Runs And Models

Experiment-specific outputs live under `runs/`.

```text
runs/
  metrics/      # metric computation runs
  training/     # training runs, logs, evaluation outputs
  analysis/     # statistical analysis runs
```

Saved model artifacts live under `models/`.

```text
models/
  checkpoints/
  reports/
```

Use timestamped or descriptive subfolders for run outputs, for example:

```text
runs/training/2026-09-20_baseline_xgboost/
```

Do not write training outputs, plots, checkpoints, metrics CSVs, or logs to the repository root.

## Documentation

Documentation lives under `docs/`.

Use `docs/` for architecture notes, metric definitions, reports, schemas, and longer explanations.

## Legacy Files

Old datasets and outputs that have not yet been classified can temporarily live under `legacy/`.
Files should move out of `legacy/` only when their role is clear:

- source data -> `data/raw/`
- external exports -> `data/external/`
- small examples -> `data/examples/`
- canonical processed outputs -> `data/processed/`
- obsolete reference material -> `notebooks/legacy/` or `docs/`

## Placement Rules

Before creating a file, choose its destination:

- reusable Python code -> `src/`
- runnable one-off or CLI script -> `scripts/`
- notebook -> `notebooks/<topic>/`
- raw source data -> `data/raw/`
- external dataset/export -> `data/external/`
- example dataset -> `data/examples/`
- intermediate generated data -> `data/interim/`
- stable processed data -> `data/processed/`
- experiment/run output -> `runs/`
- model checkpoint/report -> `models/`
- documentation -> `docs/`

If a new file does not clearly fit one of these locations, create a short note in `docs/` before adding a new top-level folder.
