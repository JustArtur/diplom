"""Общие доменные константы проекта."""

from __future__ import annotations

import numpy as np

WORLD_SIZE = 5_000_000
MIN_HEIGHT = 60.0
# Максимальная высота соответствует верхнему уровню ERA5 (~50 гПа ≈ 20 км).
MAX_HEIGHT = 20000.0
# Физически обоснованный потолок вертикальной скорости — предотвращает численный взрыв.
MAX_VERTICAL_SPEED = 50.0
# Базовая стартовая позиция — центр мира по X/Y, высота 8 км.
INITIAL_POSITION = np.array([WORLD_SIZE / 2.0, WORLD_SIZE / 2.0, 60.0], dtype=np.float32)
# Базовая целевая позиция — центр мира по X/Y, высота 10 км.
TARGET_POSITION = np.array([WORLD_SIZE / 2.0, WORLD_SIZE / 2.0, 10_000.0], dtype=np.float32)
