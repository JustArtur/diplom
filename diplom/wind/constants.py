"""Общие константы для работы с ветровым полем."""

from __future__ import annotations

from pathlib import Path

DEFAULT_WIND_DATA_PATH = Path("data/era5_sample.nc")

# Опорная точка для локальной системы координат: юго-западный угол области запроса.
DEFAULT_ORIGIN_LAT = 30.0
DEFAULT_ORIGIN_LON = 28.0
