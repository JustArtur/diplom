"""Константы среды обучения."""

import numpy as np

# Ограничение скорости накачки/откачки воздуха в баллон (кг/с).
ACTION_LIMIT = 5.0
DEFAULT_DT = 0.5
# 500 000 шагов × dt=0.5 с = примерно 3 дня симуляции на эпизод.
MAX_EPISODE_STEPS = 1_700_000
# Лимит шагов эпизода при обучении (чаще done → ep_rew_mean и success_rate в TB).
TRAIN_MAX_EPISODE_STEPS = 1_700_000
# Горизонтальный progress (Δdist_xy, м) / scale; вертикаль слабее (выбор слоя ветра).
REWARD_HORIZONTAL_PROGRESS_SCALE = 1000.0
REWARD_VERTICAL_PROGRESS_SCALE = 4000.0
# Плотный штраф −distance_xy / scale (без 3D — меньше «прыжка» только по Z).
REWARD_HORIZONTAL_DISTANCE_SCALE = 75_000.0
REWARD_ENERGY_COEF = 0.5
REWARD_ENERGY_SCALE = 100.0
REWARD_BOUNDARY_PENALTY = 0.05
SUCCESS_REWARD = 500.0
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
# Стартовые дельты куррикулума (этап 1); дальше см. TrainPositionCurriculumCallback.
TRAIN_INITIAL_POSITION_DELTA = np.array([20_000.0, 20_000.0, 5_000.0], dtype=np.float32)
TRAIN_TARGET_POSITION_DELTA = np.array([20_000.0, 20_000.0, 5_000.0], dtype=np.float32)
