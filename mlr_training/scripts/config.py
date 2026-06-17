"""Central path config. Override data location with $DNABREATHING_DATA."""
import os
from pathlib import Path

# scripts/ -> mlr_training/ (repo subroot for this pipeline)
ROOT = Path(__file__).resolve().parent.parent

# Data downloaded from OSF (gitignored). Default: <mlr_training>/data
DATA_ROOT = Path(os.environ.get("DNABREATHING_DATA", ROOT / "data"))

# --- Data inputs (from OSF) ---
INPUT_DIR        = DATA_ROOT / "ht-selex-data-revision"
SHAPE_DIR        = DATA_ROOT / "deepdnashape_results"
BREATHING_DIR    = DATA_ROOT / "pyDNA_EPBD_outputs"      # per-TF subdirs of <idx>.pkl
PREPROCESSED_DIR = DATA_ROOT / "preprocessed_data"

# --- Committed artifacts (in repo) ---
MODELS_DIR  = ROOT / "models"                 # mlr_model.py writes models/mlr/<feature_set>/
SUMMARY_DIR = ROOT / "results" / "summary"
FIGURES_DIR = ROOT / "results" / "figures"
