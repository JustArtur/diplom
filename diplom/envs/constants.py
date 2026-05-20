"""Константы среды обучения."""

import numpy as np

# Ограничение скорости накачки/откачки воздуха в баллон (кг/с).
ACTION_LIMIT = 5.0
DEFAULT_DT = 0.5
# 2 500 000 шагов × dt=0.5 с ≈ 14.5 суток симуляции на эпизод.
MAX_EPISODE_STEPS = 2_500_000
# Лимит шагов эпизода при обучении PPO (верхняя граница / eval без куррикулума).
TRAIN_MAX_EPISODE_STEPS = 2_500_000
# Куррикулум длины эпизода: 300k → … → 2.5M; длительность этапа растёт на interval_growth.
TRAIN_EPISODE_LENGTH_CURRICULUM_MIN = 300_000
TRAIN_EPISODE_LENGTH_CURRICULUM_MAX = 2_500_000
TRAIN_EPISODE_LENGTH_CURRICULUM_STEP = 300_000
TRAIN_EPISODE_LENGTH_CURRICULUM_INTERVAL = 2_000_000
TRAIN_EPISODE_LENGTH_CURRICULUM_INTERVAL_GROWTH = 500_000
# Горизонтальный progress (Δdist_xy, м) / scale; асимметричные веса — см. EnvironmentConfig.
REWARD_HORIZONTAL_PROGRESS_SCALE = 1000.0
REWARD_VERTICAL_PROGRESS_SCALE = 4000.0
# Слабый фон −distance_xy (основной сигнал — wind alignment).
REWARD_HORIZONTAL_DISTANCE_SCALE = 75_000.0
REWARD_HORIZONTAL_DISTANCE_COEF = 0.01
# Ветер к цели (м/с) / scale; Δalignment и бонус за новый минимум dist_xy.
REWARD_WIND_ALIGN_SCALE = 20.0
REWARD_WIND_ALIGN_COEF = 1.0
REWARD_WIND_ALIGN_DELTA_COEF = 0.5
# Серия шагов с попутным ветром (wind_toward > 0) / застревание (wind_toward < −5 м/с).
REWARD_WIND_FAVORABLE_THRESHOLD = 0.0
REWARD_WIND_ADVERSE_THRESHOLD = -5.0
REWARD_WIND_FAVORABLE_STREAK_STEPS = 100
REWARD_WIND_ADVERSE_STREAK_STEPS = 200
REWARD_WIND_FAVORABLE_STREAK_BONUS = 0.05
REWARD_WIND_ADVERSE_STREAK_PENALTY = 0.1
REWARD_HORIZONTAL_PROGRESS_POS_COEF = 0.5
REWARD_HORIZONTAL_PROGRESS_NEG_COEF = 0.075
REWARD_VERTICAL_PROGRESS_POS_COEF = 0.25
REWARD_VERTICAL_PROGRESS_NEG_COEF = 0.05
REWARD_BEST_DISTANCE_BONUS = 10.0
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
# Рандомизация старта только по X/Y; Z — всегда base_balloon.initial_position[2].
TRAIN_INITIAL_POSITION_DELTA = np.array([20_000.0, 20_000.0, 0.0], dtype=np.float32)
TRAIN_TARGET_POSITION_DELTA = np.array([20_000.0, 20_000.0, 5_000.0], dtype=np.float32)
