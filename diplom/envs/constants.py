"""Константы среды обучения."""

import numpy as np

# Ограничение скорости накачки/откачки воздуха в баллон (кг/с).
ACTION_LIMIT = 5.0
DEFAULT_DT = 1
# 2 500 000 шагов × dt=0.5 с ≈ 14.5 суток симуляции на эпизод (eval / верхняя граница).
MAX_EPISODE_STEPS = 2_500_000
# Лимит шагов эпизода при обучении PPO.
TRAIN_MAX_EPISODE_STEPS = 2_500_000
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
OBS_NAV_DISTANCE_SCALE = 50_000.0
# Нормализация счётчика шагов подряд с wind_toward < 0.
OBS_ADVERSE_WIND_STEPS_SCALE = 1_000.0
# Радиус успеха по XY (м).
TARGET_REACH_RADIUS = 1_000.0
# Допустимое |ΔZ| до цели при успехе (decouple XY / Z).
TARGET_VERTICAL_REACH_RADIUS = 3000.0
# Рандомизация старта только по X/Y; Z — всегда base_balloon.initial_position[2].
TRAIN_INITIAL_POSITION_DELTA = np.array([8_000.0, 8_000.0, 0.0], dtype=np.float32)
TRAIN_TARGET_POSITION_DELTA = np.array([8_000.0, 8_000.0, 1_500.0], dtype=np.float32)
