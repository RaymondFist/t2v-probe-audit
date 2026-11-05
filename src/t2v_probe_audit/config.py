"""Project paths and constants."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Inputs. Data is read-only: the analysis never writes into data/.
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = Path(os.environ.get("TPA_DATA_ROOT", DATA_DIR / "raw"))
MANIFEST = DATA_DIR / "manifest.json"

# Experiment outputs, split by kind.
OUTPUT_DIR = PROJECT_ROOT / "output"
METRICS_DIR = OUTPUT_DIR / "metrics"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"
LOG_DIR = OUTPUT_DIR / "logs"

REGIME_JSON = METRICS_DIR / "regime.json"
CONTROL_JSON = METRICS_DIR / "regime_control.json"
MULTIGEN_JSON = METRICS_DIR / "multigen.json"
NUMBERS_JSON = ANALYSIS_DIR / "numbers.json"
CHECKS_JSON = ANALYSIS_DIR / "checks.json"

# Rendered deliverables.
FIGURES_DIR = PROJECT_ROOT / "figures"
TABLES_DIR = PROJECT_ROOT / "tables"

# Working resolution for pixel and flow metrics.
WORK_W, WORK_H = 512, 288

# Embedding input resolution (ViT patch 14 divides 224).
DINO_SIZE = 224

# Farneback parameters.
FLOW_PARAMS = dict(
    pyr_scale=0.5, levels=3, winsize=15, iterations=3,
    poly_n=5, poly_sigma=1.2, flags=0,
)

# Magnitude channel scaling (tanh saturation).
LAMBDA_MAG = 1.0

# Guard thresholds.
GUARD_MIN_MAG = 0.1          # median displacement below this: no motion
PAIR_MIN_MAG = 0.2           # per-pixel magnitude below this carries no direction
DEGENERATE_INFLATION = 1.2   # reject if induced > 1.2 * observed + 0.5 px

# Metrics where a larger value is better.
HIGHER_BETTER = {
    "dino_pair_mean", "dino_pair_min", "mcs", "mcs_disc",
    "mcs_warp", "mcs_mag", "mcs_disc_guard", "mcs_warp_guard",
    "mcs_mag_guard", "mcs_guard",
}

# Generators present in the public corpus used here.
GENERATORS = ["floor33", "gen2", "pika", "modelscope", "zeroscope"]

# Two-second truncation policy, applied in seconds rather than in frames.
HORIZON_SECONDS = 2.0

DINO_REPO = Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main"
