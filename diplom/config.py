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
    TRAIN_TARGET_POSITION_HORIZONTAL_DELTA,
    TRAIN_TARGET_POSITION_VERTICAL_DELTA,
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

DEFAULT_VARIABLES: tuple[str, ...] = (
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "geopotential",
    "temperature",
)


@dataclass(frozen=True, slots=True)
class BalloonConfig:
    initial_position: np.ndarray | None = None
    target_position: np.ndarray | None = None
    sim_time: np.datetime64 | None = None


@dataclass(frozen=True, slots=True)
class DownloadConfig:
    outfile: Path = DEFAULT_ERA5_OUTFILE
    north: float = DEFAULT_ERA5_NORTH
    west: float = DEFAULT_ERA5_WEST
    south: float = DEFAULT_ERA5_SOUTH
    east: float = DEFAULT_ERA5_EAST
    start: str = DEFAULT_ERA5_START
    end: str = DEFAULT_ERA5_END
    pressure_levels: tuple[str, ...] = DEFAULT_PRESSURE_LEVELS
    variables: tuple[str, ...] = DEFAULT_VARIABLES
    hour_step: int = 24


@dataclass(frozen=True, slots=True)
class WindConfig:
    path: Path = DEFAULT_ERA5_OUTFILE
    # Опорная широта для локальной системы координат
    # Если не задана, будет взята из фактической юго-западной границы файла
    origin_lat: float | None = None
    # Опорная долгота для локальной системы координат
    # Если не задана, будет взята из фактической юго-западной границы файла
    origin_lon: float | None = None


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    balloon: BalloonConfig = field(default_factory=BalloonConfig)
    dt: float = DEFAULT_DT
    # при совместном использовании, чтобы поведение RL-среды и визуализации было идентичным
    initial_air_weight: float = DEFAULT_AIR_WEIGHT
    randomize_initial_position: bool = False
    randomize_target_position: bool = False
    train_initial_position_delta: np.ndarray = field(
        default_factory=lambda: TRAIN_INITIAL_POSITION_DELTA.copy()
    )
    train_target_position_horizontal_delta: float = TRAIN_TARGET_POSITION_HORIZONTAL_DELTA
    train_target_position_vertical_delta: float = TRAIN_TARGET_POSITION_VERTICAL_DELTA
    action_limit: float = ACTION_LIMIT
    target_reach_radius: float = TARGET_REACH_RADIUS
    max_episode_steps: int = MAX_EPISODE_STEPS
    train_max_episode_steps: int = TRAIN_MAX_EPISODE_STEPS
    normalize_observations: bool = True
    trajectory_steps_dir: Path | None = None
    # Сохранять ли observation, из которого был выбран action
    # Нужен для демонстраций и pretraining, но для обычного train обычно не требуется
    trajectory_record_observation: bool = False
    # Успешные эпизоды всегда копируются в trajectories/_success/ и с диска не удаляются
    trajectory_max_history: int = 3
    trajectory_show_wind_cones: bool = False
    trajectory_combined_html: bool = True
    reward_name: str = "simple"
    obs_name: str = "default"


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    balloon: BalloonConfig = field(default_factory=BalloonConfig)
    initial_air_weight: float = DEFAULT_AIR_WEIGHT


@dataclass(frozen=True, slots=True)
class EpisodeLengthCurriculumStage:
    # until_timesteps=None, без верхней границы (только последний этап)

    from_timesteps: int
    until_timesteps: int | None
    max_episode_steps: int


EpisodeLengthCurriculumStageInput: TypeAlias = (
    EpisodeLengthCurriculumStage | tuple[int, int | None, int]
)


DEFAULT_EPISODE_LENGTH_CURRICULUM_STAGES: tuple[EpisodeLengthCurriculumStage, ...] = (
    EpisodeLengthCurriculumStage(0, 3_000_000, 100_000),
    # EpisodeLengthCurriculumStage(3_000_000, 8_000_000, 800_000)
    # EpisodeLengthCurriculumStage(8_000_000, None, 1_250_000)
)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    total_timesteps: int = 20_000_000
    seed: int = 0
    logdir: Path = Path("ppo")
    profile_logdir: Path = Path("profile_ppo")
    # больше 8 даёт только IPC-overhead без прироста скорости
    n_envs: int = 2
    # cpu | cuda | mps
    device: str = "cpu"
    verbose: int = 1
    model_name: str = "default"
    # n_steps PPO на среду за rollout (должен совпадать с n_steps в PPO(...))
    ppo_n_steps: int = 4096
    # Энтропийный коэффициент PPO (меньше, меньше раздувается log_std)
    ent_coef: float = 0.008
    ent_coef_start: float = 0.02
    ent_coef_end: float = 0.008
    ent_coef_decay_timesteps: int = 5_000_000
    # Learning rate PPO
    learning_rate: float = 1e-4
    max_grad_norm: float = 0.3
    experiment_name: str | None = None
    # Этапы куррикулума длины эпизода: (from_ts, until_ts, max_episode_steps); until_ts=None, навсегда
    episode_length_curriculum_enabled: bool = True
    episode_length_curriculum_stages: tuple[EpisodeLengthCurriculumStageInput, ...] = (
        DEFAULT_EPISODE_LENGTH_CURRICULUM_STAGES
    )


DEFAULT_WINDOW_SIZE: tuple[int, int] = (2560, 1440)


@dataclass(frozen=True, slots=True)
class VisualizationConfig:
    # Размер окна визуализации
    window_size: tuple[int, int] = DEFAULT_WINDOW_SIZE
    # Нижний цвет фона сцены
    bg_bottom: str = "deepskyblue"
    # Верхний цвет фона сцены
    bg_top: str = "midnightblue"
    # Время старта визуализации; None, первый шаг времени из ERA5
    sim_start_time: np.datetime64 | None = None


@dataclass(frozen=True, slots=True)
class AppConfig:
    # Конфиг для скачивания данных ERA5
    download: DownloadConfig = field(default_factory=DownloadConfig)
    # Конфиг ветрового интерполятора
    wind: WindConfig = field(default_factory=WindConfig)
    # Конфиг среды RL
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    # Конфиг одиночной физической симуляции
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    # Конфиг обучения
    training: TrainingConfig = field(default_factory=TrainingConfig)
    # Конфиг визуализации
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


