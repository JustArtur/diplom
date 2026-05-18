"""Константы среды обучения."""

import numpy as np

ACTION_LIMIT = 10  # Ограничение скорости накачки воздуха в баллон
DEFAULT_DT = 0.5
# 500 000 шагов × dt=0.5 с = примерно 3 дня симуляции на эпизод.
MAX_EPISODE_STEPS = 1_000_000
# Радиус вокруг цели, при достижении которого эпизод считается успешным.
TARGET_REACH_RADIUS = 25.0
# Радиус рандомизации стартовой и целевой позиции по X/Y — 75 км,
# по высоте — 15 км.
TRAIN_INITIAL_POSITION_DELTA = np.array([75_000.0, 75_000.0, 0], dtype=np.float32)
TRAIN_TARGET_POSITION_DELTA = np.array([75_000.0, 75_000.0, 15_000.0], dtype=np.float32)