"""Central configuration: paths, column names and global settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "charging_record.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "charging_clean.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.20                 # chronological split (last 20% of sessions)

# Dataset columns
TARGET = "Demand"
CAT_BASE = ["ChargerCompany", "Location", "ChargerType"]

# Duration is only known AFTER a session ends and is almost a proxy for Demand.
# Keep False to avoid target leakage when predicting demand "ahead of time".
USE_DURATION_AS_FEATURE = False

# Classification
HIGH_DEMAND_QUANTILE = 0.5       # median threshold (computed on TRAIN only)
IMBALANCE_THRESHOLD = 0.40       # minority share below this => resample
SLOW_MODEL_MAX_ROWS = 20000      # subsample for KNN / SVM

# Time series
TS_FREQ = "D"                    # "D" daily or "H" hourly
TS_HOLDOUT = 30                  # periods held out for evaluation
TS_HORIZON = 30                  # periods forecast into the future
