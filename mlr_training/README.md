# MLR Training Pipeline

## Overview

This directory contains the multiple linear regression (MLR) training pipeline
to evaluate whether DNA breathing features (bubbles and flip) improve TF 
binding score prediction beyond information already encoded by 1mer sequence and
DNA shape features. MLR models (10-fold cross-validation) are trained for each TF 
on HT-SELEX data using different combinations of feature sets (1mer, breathing,
Deep DNAshape), and cross-validated R² values are compared across 38 TF datasets.

## Repository layout

```
mlr_training/
├── scripts/                 – Python + shell pipeline scripts
│   ├── config.py            – central path config ($DNABREATHING_DATA)
│   ├── preprocess.py        – HT-SELEX seqs → feature CSVs
│   ├── mlr_model.py         – Ridge MLR with 10-fold CV
│   ├── summary.py           – tabulate per-TF R²/MAE from saved models
│   ├── load_breathing_features.py
│   ├── plot_comparison.py / plot_comparison_multi.py
│   ├── plot_runtime.py
├── models/mlr/              – trained Ridge models (*.pkl)
│   └── <feature_set>/       – 22 feature-set subdirs, one *.pkl per TF
├── results/
│   ├── summary/             – per-feature-set R² tables (*.txt)
│   └── figures/             – publication figures fig_4/5/s7 + runtime
├── env/
│   └── requirements.txt
├── data/                    – OSF data (gitignored; see Data section below)
└── README.md
```

## Dependencies

### Python packages

```bash
pip install -r env/requirements.txt
```

Requires Python ≥ 3.9.

### External tools

These tools are used during the preprocessing stage and have their own
installation instructions and conda environments.

| Tool | Purpose | Repository |
|------|---------|------------|
| `pyDNA_EPBD` | EPBD breathing simulation (generates `pyDNA_EPBD_outputs/`) | https://github.com/lanl/pyDNA_EPBD |
| `deepDNAshape` | DNA shape prediction (generates `deepdnashape_results/`) | https://github.com/JinsenLi/deepDNAshape |

## Data

Data is stored in OSF (**DOI: 10.17605/OSF.IO/UAHBJ**) and extract so that:

```
export DNABREATHING_DATA=/path/to/data

$DNABREATHING_DATA/ht-selex-data-revision/    # 38 TF HT-SELEX datasets
$DNABREATHING_DATA/deepdnashape_results/       # shape features
$DNABREATHING_DATA/preprocessed_data/         # ready-to-train CSVs
$DNABREATHING_DATA/pyDNA_EPBD_outputs/        # EPBD breathing pkls (large)
```

If `DNABREATHING_DATA` is not set, scripts default to `mlr_training/data/`.

data.tar.gz (HT-SELEX + shape + preprocessed CSVs) is sufficient to train and
evaluate models without the pyDNA_EPBD outputs.

## Reproduce

All commands are run from the `mlr_training/` directory.

### Train models + generate summaries

```bash
export DNABREATHING_DATA=/path/to/data   # or omit to use mlr_training/data/

# Train a single feature set
python scripts/mlr_model.py --all "${DNABREATHING_DATA:-data}/preprocessed_data" \
    --features 1mer breathing deepdnashape

# Summarize results
python scripts/summary.py models/mlr/1mer+breathing+deepdnashape \
    > results/summary/1mer+breathing+deepdnashape.txt
```

### Run all 11 feature sets

```bash
sbatch scripts/run_model.sh 
```

### Generate figures

```bash
# Publication figures (fig 4, 5, S7)
bash scripts/run_plot_comparison_multi.sh

# Runtime comparison figure
python scripts/plot_runtime.py

# Individual pairwise scatter
python scripts/plot_comparison.py results/summary/1mer.txt results/summary/1mer+breathing.txt
```

### Generate features from scratch (optional)

1. Run `pyDNA_EPBD` to produce EPBD breathing pkls in `pyDNA_EPBD_outputs/`.
2. Run `bash scripts/run_deepdnashape.sh` to produce shape features in
   `deepdnashape_results/`.
3. `python scripts/preprocess.py --all` to generate `preprocessed_data/*.csv`.

