from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from diplom.envs.constants import (
    ACTION_LIMIT,
    DEFAULT_DT,
    MAX_EPISODE_STEPS,
    REWARD_BOUNDARY_PENALTY,
    REWARD_ENERGY_COEF,
    REWARD_ENERGY_SCALE,
    REWARD_HORIZONTAL_DISTANCE_SCALE,
    REWARD_HORIZONTAL_PROGRESS_SCALE,
    REWARD_VERTICAL_PROGRESS_SCALE,
    SUCCESS_REWARD,
    TARGET_REACH_RADIUS,
    TRAIN_INITIAL_POSITION_DELTA,
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
    '5',
    '3',
    '1'
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
    # Максимальное число шагов в одном эпизоде (eval/rollout; в train подменяется train_max_episode_steps).
    max_episode_steps: int = MAX_EPISODE_STEPS
    # Лимит шагов эпизода при обучении PPO.
    train_max_episode_steps: int = TRAIN_MAX_EPISODE_STEPS
    # progress_xy / scale; progress_z слабее; штраф −distance_xy; штраф у границы мира.
    reward_horizontal_progress_scale: float = REWARD_HORIZONTAL_PROGRESS_SCALE
    reward_vertical_progress_scale: float = REWARD_VERTICAL_PROGRESS_SCALE
    reward_horizontal_distance_scale: float = REWARD_HORIZONTAL_DISTANCE_SCALE
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
    # Каталог для чекпоинтов и метрик обучения (run-ы: PPO_0/tb, PPO_0/trajectories, …).
    logdir: Path = Path("runs/ppo")
    # Каталог run-ов при profile-ppo-mem / profile-ppo-cpu.
    profile_logdir: Path = Path("runs/profile_ppo")
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
    ent_coef: float = 0.02
    # Learning rate PPO.
    learning_rate: float = 1e-4
    # Ограничение нормы градиента (стабильность при всплесках KL).
    max_grad_norm: float = 0.3
    # По мере timesteps расширять окно рандомизации старта/цели (см. curriculum_callback).
    curriculum_enabled: bool = True


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


