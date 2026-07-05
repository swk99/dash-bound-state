# config.py
import os
from pathlib import Path

# =============================================================================
# 1. PROJECT INFRASTRUCTURE & STORAGE
# =============================================================================
BASE_DIR = Path(__file__).parent
ART_DIR = BASE_DIR / "artifacts"
FIG_DIR = BASE_DIR / "research_figures"

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

LABELED_PATH = DATA_DIR / "labeled_train.parquet" #how to save the labeling.py's data

for folder in [ART_DIR, FIG_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123")
BUCKET_NAME = "cold-storage"
S3_DATA_PREFIX = "df1s_flush/"

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# =============================================================================
# 2. SYSTEM MODEL & PARAMETERS
# =============================================================================
SYMBOL = "btcusdt_1"

# EWMA
LAMBDA = 0.94
EPSILON = 1e-9

# Shock labeling (H is ticks/events)
HORIZON_H = 30
ALPHA = 2.0

# Two-stage decision
TAU_CONF = 0.6731 #Decided by offline_dataset test
LOOKBACK_W = 30

FEATURE_COLS = ["r_t", "sigma_hat", "OFI_t", "Imbalance_t", "VolSpike_t", "msg_count"]

# -------------------------
# Imbalance calibration knobs (optional)
# -------------------------
USE_TEMP_SCALE = False
TEMP_NEG = 1.0
TEMP_POS = 1.0

USE_LDAM = False
LDAM_C = 0.0

USE_LOGIT_ADJ = False
LA_TAU = 1.0

# =============================================================================
# 2.1 Tag formatting (SINGLE source of truth)
# =============================================================================
def make_tag(
    h: int = HORIZON_H,
    alpha: float = ALPHA,
    lambda_val: float = LAMBDA,
) -> str:
    """
    Canonical artifact tag used everywhere.
    - H: int ticks/events
    - alpha: compact float (3.0 -> 3, 1.5 -> 1.5)
    - lambda: stored as integer percent (0.94 -> 94)
    """
    l_int = int(round(lambda_val * 100))
    return f"H{int(h)}_a{alpha:g}_L{l_int:02d}"

MODEL_TAG = make_tag()

# =============================================================================
# 3. EXPERIMENT & SENSITIVITY ANALYSIS SWEEP
# =============================================================================
N_STEPS = 5000
K_LIST = [1, 5, 10, 20, 50]

