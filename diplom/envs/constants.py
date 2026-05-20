"""Константы среды обучения."""

import numpy as np

ACTION_LIMIT = 15  # Ограничение скорости накачки/откачки воздуха в баллон (кг/с)
DEFAULT_DT = 0.5
# 500 000 шагов × dt=0.5 с = примерно 3 дня симуляции на эпизод.
MAX_EPISODE_STEPS = 1_000_000
# Лимит шагов эпизода при обучении (чаще done → ep_rew_mean и success_rate в TB).
TRAIN_MAX_EPISODE_STEPS = 1_000_000
# Компоненты reward: progress (Δdist), −distance, штраф за энергию за шаг.
REWARD_PROGRESS_SCALE = 1000.0
REWARD_DISTANCE_SCALE = 75_000.0
REWARD_ENERGY_COEF = 0.01
REWARD_ENERGY_SCALE = 100.0
SUCCESS_REWARD = 1.0
# Масштабы для нормализации наблюдений (фиксированные, совместимы с worker rollout).
OBS_XY_SCALE = 75_000.0
OBS_ALTITUDE_SCALE = 15_000.0
OBS_WIND_SCALE = 50.0
OBS_VERTICAL_SPEED_SCALE = 10.0
OBS_VERTICAL_ACCELERATION_SCALE = 2.0
OBS_ENERGY_SCALE = 1_000_000.0
OBS_AIR_WEIGHT_SCALE = 1_000.0
OBS_AIR_DENSITY_SCALE = 2.0
OBS_TEMPERATURE_SCALE = 300.0
OBS_PRESSURE_SCALE = 100_000.0
# Радиус вокруг цели, при достижении которого эпизод считается успешным.
TARGET_REACH_RADIUS = 25.0
# Радиус рандомизации стартовой и целевой позиции по X/Y — 75 км,
# по высоте — 15 км.
TRAIN_INITIAL_POSITION_DELTA = np.array([75_000.0, 75_000.0, 0], dtype=np.float32)
TRAIN_TARGET_POSITION_DELTA = np.array([75_000.0, 75_000.0, 15_000.0], dtype=np.float32)