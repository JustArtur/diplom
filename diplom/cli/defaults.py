"""Дефолты CLI, не дублирующие доменную логику config."""

from __future__ import annotations

from pathlib import Path

from diplom.config import DownloadConfig, TrainingConfig
from diplom.data.era5_paths import DEFAULT_ERA5_OUTFILE, era5_dataset_title

DEFAULT_DOWNLOAD_CONFIG = DownloadConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_ROLLOUT_MODEL_PATH = (
    DEFAULT_TRAINING_CONFIG.logdir
    / era5_dataset_title(DEFAULT_ERA5_OUTFILE)
    / "ppo_model.zip"
)
