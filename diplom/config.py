from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from diplom.envs.constants import (
    ACTION_LIMIT,
    DEFAULT_DT,
    MAX_EPISODE_STEPS,
    REWARD_BEST_DISTANCE_BONUS,
    REWARD_BOUNDARY_PENALTY,
    REWARD_ENERGY_COEF,
    REWARD_ENERGY_SCALE,
    REWARD_HIGH_ALTITUDE_ADVERSE_PENALTY,
    REWARD_HIGH_ALTITUDE_M,
    REWARD_HORIZONTAL_DISTANCE_COEF,
    REWARD_HORIZONTAL_DISTANCE_SCALE,
    REWARD_HORIZONTAL_PROGRESS_NEG_COEF,
    REWARD_HORIZONTAL_PROGRESS_POS_COEF,
    REWARD_HORIZONTAL_PROGRESS_SCALE,
    REWARD_IDLE_ACTION_MIN_DZ_M,
    REWARD_IDLE_ACTION_PENALTY,
    REWARD_IDLE_ACTION_STREAK_STEPS,
    REWARD_IDLE_ACTION_THRESHOLD,
    REWARD_VERTICAL_PROGRESS_NEG_COEF,
    REWARD_VERTICAL_PROGRESS_POS_COEF,
    REWARD_VERTICAL_PROGRESS_SCALE,
    REWARD_WIND_ALIGN_ADVERSE_PROGRESS_SCALE,
    REWARD_WIND_ALIGN_COEF,
    REWARD_WIND_ALIGN_DELTA_COEF,
    REWARD_WIND_ALIGN_SCALE,
    REWARD_WIND_ALIGN_ZERO_PROGRESS_STEPS,
    REWARD_WIND_ADVERSE_STREAK_PENALTY,
    REWARD_WIND_ADVERSE_STREAK_STEPS,
    REWARD_WIND_ADVERSE_THRESHOLD,
    REWARD_WIND_FAVORABLE_STREAK_BONUS,
    REWARD_WIND_FAVORABLE_STREAK_STEPS,
    REWARD_WIND_FAVORABLE_THRESHOLD,
    REWARD_WIND_SCAN_DELTA_COEF,
    REWARD_WIND_SCAN_MIN_DZ_M,
    REWARD_Z_STICK_MIN_STD_M,
    REWARD_Z_STICK_PENALTY,
    REWARD_Z_STICK_WINDOW_STEPS,
    SUCCESS_REWARD,
    TARGET_REACH_RADIUS,
    TARGET_VERTICAL_REACH_RADIUS,
    TRAIN_INITIAL_POSITION_DELTA,
    TRAIN_EPISODE_LENGTH_CURRICULUM_INTERVAL,
    TRAIN_EPISODE_LENGTH_CURRICULUM_INTERVAL_GROWTH,
    TRAIN_EPISODE_LENGTH_CURRICULUM_MAX,
    TRAIN_EPISODE_LENGTH_CURRICULUM_MIN,
    TRAIN_EPISODE_LENGTH_CURRICULUM_STEP,
    TRAIN_MAX_EPISODE_STEPS,
    TRAIN_TARGET_POSITION_DELTA,
)
from diplom.sim.constants import DEFAULT_AIR_WEIGHT
from diplom.viz.constants import WINDOW_SIZE
from diplom.data.era5_paths import (
    DEFAULT_ERA5_EAST,
    DEFAULT_ERA5_END,
    DEFAULT_ERA5_NORTH,
    DEFAULT_ERA5_OUTFILE,
    DEFAULT_ERA5_SOUTH,
    DEFAULT_ERA5_START,
    DEFAULT_ERA5_WEST,
)


# Набор уровней давления ERA5, которые используем для обучения и симуляции по умолчанию.
DEFAULT_PRESSURE_LEVELS: tuple[str, ...] = (
    '1000',
    '900',
    '800',
    '700',
    '650',
    '600',
    '550',
    '500',
    '450',
    '400',
    '350',
    '300',
    '250',
    '200',
    '150',
    '100',
    '90',
    '80',
    '70',
    '60',
    '50',
    '40',
    '30',
    '25',
    '20',
    '15',
    '10',
)

# Набор переменных ERA5, нужных для расчёта ветра, температуры и вертикального движения.
DEFAULT_VARIABLES: tuple[str, ...] = (
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "geopotential",
    "temperature",
)


@dataclass(frozen=True, slots=True)
class BalloonConfig:
    # Базовая стартовая позиция аэростата в локальной системе координат.
    # Если не задана, будет подставлена из географических границ датасета.
    initial_position: np.ndarray | None = None
    # Базовая целевая точка, к которой должен стремиться агент.
    # Если не задана, будет подставлена из географических границ датасета.
    target_position: np.ndarray | None = None
    # Стартовое модельное время для запроса ветра; None — первый шаг времени из ERA5.
    sim_time: np.datetime64 | None = None


@dataclass(frozen=True, slots=True)
class DownloadConfig:
    # Куда сохранять скачанный ERA5-файл.
    outfile: Path = DEFAULT_ERA5_OUTFILE
    # Северная граница области запроса.
    north: float = DEFAULT_ERA5_NORTH
    # Западная граница области запроса.
    west: float = DEFAULT_ERA5_WEST
    # Южная граница области запроса.
    south: float = DEFAULT_ERA5_SOUTH
    # Восточная граница области запроса.
    east: float = DEFAULT_ERA5_EAST
    # Начало периода скачивания.
    start: str = DEFAULT_ERA5_START
    # Конец периода скачивания.
    end: str = DEFAULT_ERA5_END
    # Уровни давления, которые нужно запросить.
    pressure_levels: tuple[str, ...] = DEFAULT_PRESSURE_LEVELS
    # Список переменных ERA5, которые нужно скачать.
    variables: tuple[str, ...] = DEFAULT_VARIABLES
    # Шаг по часам в запросе CDS: 1 — все 24 ч, 2 — 00:00, 02:00, …, 22:00.
    hour_step: int = 24


@dataclass(frozen=True, slots=True)
class WindConfig:
    # Путь к NetCDF-файлу с данными ветра.
    path: Path = DEFAULT_ERA5_OUTFILE
    # Опорная широта для локальной системы координат.
    # Если не задана, будет взята из фактической юго-западной границы файла.
    origin_lat: float | None = None
    # Опорная долгота для локальной системы координат.
    # Если не задана, будет взята из фактической юго-западной границы файла.
    origin_lon: float | None = None


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    # Базовый набор координат и времени, от которого строится эпизод среды.
    balloon: BalloonConfig = field(default_factory=BalloonConfig)
    # Шаг интегрирования среды.
    dt: float = DEFAULT_DT
    # Начальная масса воздуха в баллоне (кг); должна совпадать с SimulationConfig.initial_air_weight
    # при совместном использовании, чтобы поведение RL-среды и визуализации было идентичным.
    initial_air_weight: float = DEFAULT_AIR_WEIGHT
    # Включать ли рандомизацию стартовой позиции и цели в train-режиме.
    randomize_start_state: bool = False
    # Включать ли рандомизацию стартового времени в train-режиме.
    randomize_start_time: bool = False
    # Ширина окна случайного стартового времени от начала диапазона датасета.
    train_start_time_delta: np.timedelta64 = np.timedelta64(12, "h")
    # Амплитуда случайного смещения стартовой позиции для train-эпизодов.
    train_initial_position_delta: np.ndarray = field(
        default_factory=lambda: TRAIN_INITIAL_POSITION_DELTA.copy()
    )
    # Амплитуда случайного смещения целевой позиции для train-эпизодов.
    train_target_position_delta: np.ndarray = field(
        default_factory=lambda: TRAIN_TARGET_POSITION_DELTA.copy()
    )
    # Ограничение на модуль управляющего воздействия.
    action_limit: float = ACTION_LIMIT
    # Радиус вокруг цели, при попадании в который эпизод считается завершённым успешно.
    target_reach_radius: float = TARGET_REACH_RADIUS
    # Допустимое |ΔZ| до цели при успехе (XY и Z проверяются отдельно).
    target_vertical_reach_radius: float = TARGET_VERTICAL_REACH_RADIUS
    # Максимальное число шагов в одном эпизоде (eval/rollout; в train подменяется train_max_episode_steps).
    max_episode_steps: int = MAX_EPISODE_STEPS
    # Лимит шагов эпизода при обучении PPO.
    train_max_episode_steps: int = TRAIN_MAX_EPISODE_STEPS
    # wind_toward / scale; Δalignment; асимметричный progress_xy; слабый −distance_xy.
    reward_wind_align_scale: float = REWARD_WIND_ALIGN_SCALE
    reward_wind_align_coef: float = REWARD_WIND_ALIGN_COEF
    reward_wind_align_delta_coef: float = REWARD_WIND_ALIGN_DELTA_COEF
    reward_wind_favorable_threshold: float = REWARD_WIND_FAVORABLE_THRESHOLD
    reward_wind_adverse_threshold: float = REWARD_WIND_ADVERSE_THRESHOLD
    reward_wind_favorable_streak_steps: int = REWARD_WIND_FAVORABLE_STREAK_STEPS
    reward_wind_adverse_streak_steps: int = REWARD_WIND_ADVERSE_STREAK_STEPS
    reward_wind_favorable_streak_bonus: float = REWARD_WIND_FAVORABLE_STREAK_BONUS
    reward_wind_adverse_streak_penalty: float = REWARD_WIND_ADVERSE_STREAK_PENALTY
    reward_wind_align_adverse_progress_scale: float = REWARD_WIND_ALIGN_ADVERSE_PROGRESS_SCALE
    reward_wind_align_zero_progress_steps: int = REWARD_WIND_ALIGN_ZERO_PROGRESS_STEPS
    reward_high_altitude_m: float = REWARD_HIGH_ALTITUDE_M
    reward_high_altitude_adverse_penalty: float = REWARD_HIGH_ALTITUDE_ADVERSE_PENALTY
    reward_idle_action_threshold: float = REWARD_IDLE_ACTION_THRESHOLD
    reward_idle_action_min_dz_m: float = REWARD_IDLE_ACTION_MIN_DZ_M
    reward_idle_action_streak_steps: int = REWARD_IDLE_ACTION_STREAK_STEPS
    reward_idle_action_penalty: float = REWARD_IDLE_ACTION_PENALTY
    reward_wind_scan_min_dz_m: float = REWARD_WIND_SCAN_MIN_DZ_M
    reward_wind_scan_delta_coef: float = REWARD_WIND_SCAN_DELTA_COEF
    reward_z_stick_window_steps: int = REWARD_Z_STICK_WINDOW_STEPS
    reward_z_stick_min_std_m: float = REWARD_Z_STICK_MIN_STD_M
    reward_z_stick_penalty: float = REWARD_Z_STICK_PENALTY
    reward_horizontal_progress_scale: float = REWARD_HORIZONTAL_PROGRESS_SCALE
    reward_horizontal_progress_pos_coef: float = REWARD_HORIZONTAL_PROGRESS_POS_COEF
    reward_horizontal_progress_neg_coef: float = REWARD_HORIZONTAL_PROGRESS_NEG_COEF
    reward_vertical_progress_scale: float = REWARD_VERTICAL_PROGRESS_SCALE
    reward_vertical_progress_pos_coef: float = REWARD_VERTICAL_PROGRESS_POS_COEF
    reward_vertical_progress_neg_coef: float = REWARD_VERTICAL_PROGRESS_NEG_COEF
    reward_horizontal_distance_scale: float = REWARD_HORIZONTAL_DISTANCE_SCALE
    reward_horizontal_distance_coef: float = REWARD_HORIZONTAL_DISTANCE_COEF
    reward_best_distance_bonus: float = REWARD_BEST_DISTANCE_BONUS
    reward_energy_coef: float = REWARD_ENERGY_COEF
    reward_energy_scale: float = REWARD_ENERGY_SCALE
    reward_boundary_penalty: float = REWARD_BOUNDARY_PENALTY
    success_reward: float = SUCCESS_REWARD
    # Делить компоненты obs на фиксированные масштабы (совместимо с worker rollout).
    normalize_observations: bool = True
    # Каталог JSONL-шагов для HTML-траекторий; None — не писать шаги на диск.
    trajectory_steps_dir: Path | None = None
    # Сколько завершённых эпизодов хранить на диске для одной среды (старые удаляются).
    trajectory_max_history: int = 3


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    # Базовый набор координат и времени для одиночной симуляции.
    balloon: BalloonConfig = field(default_factory=BalloonConfig)
    # Начальная масса воздуха в баллоне.
    initial_air_weight: float = DEFAULT_AIR_WEIGHT


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    # Общее число шагов обучения PPO.
    total_timesteps: int = 4_000_000
    # Seed для воспроизводимости.
    seed: int = 0
    # Родительский каталог run-ов; фактический путь — {logdir}/{имя_датасета}/.
    logdir: Path = Path("ppo")
    # Родительский каталог для profile-ppo-mem / profile-ppo-cpu.
    profile_logdir: Path = Path("profile_ppo")
    # Количество параллельных сред. M1 Pro имеет 10 ядер (8P+2E),
    # больше 8 даёт только IPC-overhead без прироста скорости.
    n_envs: int = 8
    # Устройство для нейросети PPO: cpu | cuda | mps (см. diplom.torch_device.resolve_torch_device).
    device: str = "cpu"
    # Уровень логирования PPO в консоль (как verbose в Stable-Baselines3): 0 — тихо, 1 — таблица метрик.
    verbose: int = 1
    # Гибрид: policy+env в subprocess, rollout через shared memory (n_envs > 1).
    use_worker_policy_rollout: bool = True
    # n_steps PPO на среду за rollout (должен совпадать с n_steps в PPO(...)).
    ppo_n_steps: int = 4096
    # Энтропийный коэффициент PPO (меньше — меньше раздувается log_std).
    ent_coef: float = 0.008
    # Learning rate PPO.
    learning_rate: float = 1e-4
    # Ограничение нормы градиента (стабильность при всплесках KL).
    max_grad_norm: float = 0.3
    # По мере timesteps расширять окно рандомизации старта/цели (см. curriculum_callback).
    curriculum_enabled: bool = True
    # По мере timesteps увеличивать max_episode_steps (300k +300k до 2.5M).
    episode_length_curriculum_enabled: bool = True
    episode_length_curriculum_min: int = TRAIN_EPISODE_LENGTH_CURRICULUM_MIN
    episode_length_curriculum_max: int = TRAIN_EPISODE_LENGTH_CURRICULUM_MAX
    episode_length_curriculum_step: int = TRAIN_EPISODE_LENGTH_CURRICULUM_STEP
    episode_length_curriculum_interval: int = TRAIN_EPISODE_LENGTH_CURRICULUM_INTERVAL
    episode_length_curriculum_interval_growth: int = TRAIN_EPISODE_LENGTH_CURRICULUM_INTERVAL_GROWTH
    # Держать max_episode_steps на min, пока не будет хотя бы одного success.
    episode_length_freeze_until_success: bool = False


@dataclass(frozen=True, slots=True)
class VisualizationConfig:
    # Размер окна визуализации.
    window_size: tuple[int, int] = WINDOW_SIZE
    # Нижний цвет фона сцены.
    bg_bottom: str = "deepskyblue"
    # Верхний цвет фона сцены.
    bg_top: str = "midnightblue"
    # Время старта визуализации; None — первый шаг времени из ERA5.
    sim_start_time: np.datetime64 | None = None


@dataclass(frozen=True, slots=True)
class AppConfig:
    # Конфиг для скачивания данных ERA5.
    download: DownloadConfig = field(default_factory=DownloadConfig)
    # Конфиг ветрового интерполятора.
    wind: WindConfig = field(default_factory=WindConfig)
    # Конфиг среды RL.
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    # Конфиг одиночной физической симуляции.
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    # Конфиг обучения.
    training: TrainingConfig = field(default_factory=TrainingConfig)
    # Конфиг визуализации.
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


