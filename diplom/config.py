from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

import numpy as np

from diplom.envs.constants import (
    ACTION_LIMIT,
    DEFAULT_DT,
    MAX_EPISODE_STEPS,
    TARGET_REACH_RADIUS,
    TRAIN_INITIAL_POSITION_DELTA,
    TRAIN_MAX_EPISODE_STEPS,
    TRAIN_TARGET_POSITION_DELTA,
)
from diplom.sim.constants import DEFAULT_AIR_WEIGHT
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
    # Делить компоненты obs на фиксированные масштабы (совместимо с worker rollout).
    normalize_observations: bool = True
    # Каталог JSONL-шагов для HTML-траекторий; None — не писать шаги на диск.
    trajectory_steps_dir: Path | None = None
    # Сколько завершённых эпизодов хранить на диске для одной среды (старые удаляются).
    trajectory_max_history: int = 3
    # Показывать конусы ветра на HTML-графике траекторий.
    trajectory_show_wind_cones: bool = False
    # Имя reward-функции из diplom.envs.rewards (флаг CLI --reward).
    reward_name: str = "default"
    # Имя obs-модели из diplom.envs.observations (флаг CLI --obs).
    obs_name: str = "default"


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    # Базовый набор координат и времени для одиночной симуляции.
    balloon: BalloonConfig = field(default_factory=BalloonConfig)
    # Начальная масса воздуха в баллоне.
    initial_air_weight: float = DEFAULT_AIR_WEIGHT


@dataclass(frozen=True, slots=True)
class EpisodeLengthCurriculumStage:
    """Этап куррикулума длины эпизода.

    Активен, пока ``from_timesteps <= num_timesteps < until_timesteps``.
    ``until_timesteps=None`` — без верхней границы (только последний этап).
    """

    from_timesteps: int
    until_timesteps: int | None
    max_episode_steps: int


EpisodeLengthCurriculumStageInput: TypeAlias = (
    EpisodeLengthCurriculumStage | tuple[int, int | None, int]
)


DEFAULT_EPISODE_LENGTH_CURRICULUM_STAGES: tuple[EpisodeLengthCurriculumStage, ...] = (
    EpisodeLengthCurriculumStage(0, 7_000_000, 300_000),
    EpisodeLengthCurriculumStage(7_000_000, 17_000_000, 600_000),
    EpisodeLengthCurriculumStage(17_000_000, 30_000_000, 900_000),
    EpisodeLengthCurriculumStage(30_000_000, None, 1_250_000),
)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    # Общее число шагов обучения PPO.
    total_timesteps: int = 20_000_000
    # Seed для воспроизводимости.
    seed: int = 0
    # Родительский каталог run-ов; фактический путь — {logdir}/{experiment_name|датасет}/.
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
    # Имя PPO-модели из diplom.rl.ppo.models (флаг CLI --model).
    model_name: str = "default"
    # n_steps PPO на среду за rollout (должен совпадать с n_steps в PPO(...)).
    ppo_n_steps: int = 4096
    # Энтропийный коэффициент PPO (меньше — меньше раздувается log_std).
    ent_coef: float = 0.008
    ent_coef_start: float = 0.02
    ent_coef_end: float = 0.008
    ent_coef_decay_timesteps: int = 5_000_000
    # Learning rate PPO.
    learning_rate: float = 1e-4
    # Ограничение нормы градиента (стабильность при всплесках KL).
    max_grad_norm: float = 0.3
    # Имя эксперимента (каталог под logdir); None — stem NetCDF-датасета (CLI --experiment).
    experiment_name: str | None = None
    # Этапы куррикулума длины эпизода: (from_ts, until_ts, max_episode_steps); until_ts=None — навсегда.
    episode_length_curriculum_enabled: bool = True
    episode_length_curriculum_stages: tuple[EpisodeLengthCurriculumStageInput, ...] = (
        DEFAULT_EPISODE_LENGTH_CURRICULUM_STAGES
    )


DEFAULT_WINDOW_SIZE: tuple[int, int] = (2560, 1440)


@dataclass(frozen=True, slots=True)
class VisualizationConfig:
    # Размер окна визуализации.
    window_size: tuple[int, int] = DEFAULT_WINDOW_SIZE
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


