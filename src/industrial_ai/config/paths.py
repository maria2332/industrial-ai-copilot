"""Filesystem locations used across the project.

Paths are resolved from the repository root. Set the INDUSTRIAL_AI_ROOT environment
variable to override it (e.g. when the package is installed outside the repository).
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("INDUSTRIAL_AI_ROOT", Path(__file__).resolve().parents[3]))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_CMAPSS_DIR = RAW_DIR / "cmapss"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
