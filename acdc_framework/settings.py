from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ACDC_TRAIN_DIR = REPO_ROOT / "ACDC" / "ACDC" / "database" / "training"
ACDC_TEST_DIR = REPO_ROOT / "ACDC" / "ACDC" / "database" / "testing"
SEG_MAMBA_ROOT = REPO_ROOT / "SegMamba-V2-main"

RUNS_ROOT = REPO_ROOT / "acdc_runs"
SHARED_PREPROCESSED_DIR = RUNS_ROOT / "shared_preprocessed"
NNUNET_RAW_DIR = REPO_ROOT / "nnUNet_raw" / "Dataset701_ACDC"

DEFAULT_TARGET_SHAPE = (16, 192, 192)  # D, H, W
DEFAULT_VAL_RATIO = 0.2
NUM_CLASSES = 4
FOREGROUND_LABELS = (1, 2, 3)
RV_LABEL = 1
MYO_LABEL = 2
LV_LABEL = 3


def model_run_dir(model_name: str) -> Path:
    return RUNS_ROOT / model_name
