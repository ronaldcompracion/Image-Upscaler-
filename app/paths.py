from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "Local AI Image Enhancer"
APP_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "LocalAIImageEnhancer"
MODEL_DIR = APP_DIR / "models"
LOG_DIR = APP_DIR / "logs"
DEFAULT_OUTPUT_DIR = Path.home() / "Pictures" / "Enhanced"


def ensure_app_directories() -> None:
    for directory in (MODEL_DIR, LOG_DIR, DEFAULT_OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
