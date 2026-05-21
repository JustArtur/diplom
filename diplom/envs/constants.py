"""Константы среды обучения."""

import numpy as np

# Ограничение скорости накачки/откачки воздуха в баллон (кг/с).
ACTION_LIMIT = 5.0
DEFAULT_DT = 0.5
# 2 500 000 шагов × dt=0.5 с ≈ 14.5 суток симуляции на эпизод (eval / верхняя граница).
MAX_EPISODE_STEPS = 2_500_000
# Лимит шагов эпизода при обучении PPO.
TRAIN_MAX_EPISODE_STEPS = 2_500_000
# Куррикулум длины эпизода: 300k → … → 2.5M; длительность этапа растёт на interval_growth.
TRAIN_EPISODE_LENGTH_CURRICULUM_MIN = 300_000
TRAIN_EPISODE_LENGTH_CURRICULUM_MAX = 2_500_000
TRAIN_EPISODE_LENGTH_CURRICULUM_STEP = 300_000
TRAIN_EPISODE_LENGTH_CURRICULUM_INTERVAL = 2_000_000
TRAIN_EPISODE_LENGTH_CURRICULUM_INTERVAL_GROWTH = 8_000_000
# Горизонтальный progress (Δdist_xy, м) / scale; асимметричные веса — см. EnvironmentConfig.
REWARD_HORIZONTAL_PROGRESS_SCALE = 500.0
REWARD_VERTICAL_PROGRESS_SCALE = 4000.0
# Слабый фон −distance_xy (основной сигнал — progress + wind alignment).
REWARD_HORIZONTAL_DISTANCE_SCALE = 75_000.0
REWARD_HORIZONTAL_DISTANCE_COEF = 0.01
# Ветер к цели (м/с) / scale; Δalignment и бонус за новый минимум dist_xy.
REWARD_WIND_ALIGN_SCALE = 20.0
REWARD_WIND_ALIGN_COEF = 0.5
REWARD_WIND_ALIGN_DELTA_COEF = 0.25
# Множитель wind_align при horizontal_progress < 0 на текущем шаге.
REWARD_WIND_ALIGN_ADVERSE_PROGRESS_SCALE = 0.25
# Обнулить wind_align после стольких шагов подряд с horizontal_progress < 0.
REWARD_WIND_ALIGN_ZERO_PROGRESS_STEPS = 5
# Серия шагов с попутным ветром (wind_toward > 0) / застревание (wind_toward < −5 м/с).
REWARD_WIND_FAVORABLE_THRESHOLD = 0.0
REWARD_WIND_ADVERSE_THRESHOLD = -5.0
REWARD_WIND_FAVORABLE_STREAK_STEPS = 40
REWARD_WIND_ADVERSE_STREAK_STEPS = 200
REWARD_WIND_FAVORABLE_STREAK_BONUS = 0.08
REWARD_WIND_ADVERSE_STREAK_PENALTY = 0.1
REWARD_HORIZONTAL_PROGRESS_POS_COEF = 2.0
REWARD_HORIZONTAL_PROGRESS_NEG_COEF = 0.4
REWARD_VERTICAL_PROGRESS_POS_COEF = 0.25
REWARD_VERTICAL_PROGRESS_NEG_COEF = 0.05
REWARD_BEST_DISTANCE_BONUS = 50.0
REWARD_ENERGY_COEF = 0.5
REWARD_ENERGY_SCALE = 100.0
REWARD_BOUNDARY_PENALTY = 0.05
# Штраф за полёт выше порога во встречном ветре.
REWARD_HIGH_ALTITUDE_M = 5000.0
REWARD_HIGH_ALTITUDE_ADVERSE_PENALTY = 0.05
# Штраф за «холостое» действие: только после streak шагов с |action|≥порога и |dz|<min_dz.
REWARD_IDLE_ACTION_THRESHOLD = 0.3
REWARD_IDLE_ACTION_MIN_DZ_M = 0.1
REWARD_IDLE_ACTION_STREAK_STEPS = 40
REWARD_IDLE_ACTION_PENALTY = 0.02
# Бонус за улучшение wind_toward после смены слоя (|Δz| ≥ порога).
REWARD_WIND_SCAN_MIN_DZ_M = 50.0
REWARD_WIND_SCAN_DELTA_COEF = 1.0
# Штраф за залипание по высоте: std(z) за окно < min_std.
REWARD_Z_STICK_WINDOW_STEPS = 50_000
REWARD_Z_STICK_MIN_STD_M = 200.0
REWARD_Z_STICK_PENALTY = 0.03
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
# wind_toward на фиксированных высотах (м) — «карта слоёв» для политики.
OBS_PROBE_ALTITUDES_M: tuple[float, ...] = (500.0, 3000.0, 10_000.0)
OBS_DIM = 20 + len(OBS_PROBE_ALTITUDES_M)
# Радиус вокруг цели по XY, при достижении которого эпизод считается успешным.
TARGET_REACH_RADIUS = 25.0
# Допустимое |ΔZ| до цели при успехе (decouple XY / Z).
TARGET_VERTICAL_REACH_RADIUS = 3000.0
# Рандомизация старта только по X/Y; Z — всегда base_balloon.initial_position[2].
TRAIN_INITIAL_POSITION_DELTA = np.array([12_000.0, 12_000.0, 0.0], dtype=np.float32)
TRAIN_TARGET_POSITION_DELTA = np.array([12_000.0, 12_000.0, 1_500.0], dtype=np.float32)
