from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from diplom.envs.constants import (
    ACTION_LIMIT,
    DEFAULT_DT,
    MAX_EPISODE_STEPS,
    TARGET_REACH_RADIUS,
    TRAIN_INITIAL_POSITION_DELTA,
    TRAIN_TARGET_POSITION_DELTA,
)
from diplom.sim.constants import DEFAULT_AIR_WEIGHT, SIM_TIME
from diplom.viz.constants import WINDOW_SIZE
from diplom.wind.constants import DEFAULT_WIND_DATA_PATH


# Набор уровней давления ERA5, которые используем для обучения и симуляции по умолчанию.
DEFAULT_PRESSURE_LEVELS: tuple[str, ...] = (
    '1000',
    '900',
    '800',
    '700'
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
    '80'
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
    # Стартовое модельное время, которое используется для запроса ветра.
    sim_time: np.datetime64 = SIM_TIME


@dataclass(frozen=True, slots=True)
class DownloadConfig:
    # Куда сохранять скачанный ERA5-файл.
    outfile: Path = Path("data/era5_sample.nc")
    # Северная граница области запроса.
    north: float = 64.0
    # Западная граница области запроса.
    west: float = 45.0
    # Южная граница области запроса.
    south: float = 47.0
    # Восточная граница области запроса.
    east: float = 62.0
    # Начало периода скачивания.
    start: str = "2024-07-01"
    # Конец периода скачивания.
    end: str = "2024-07-03"
    # Уровни давления, которые нужно запросить.
    pressure_levels: tuple[str, ...] = DEFAULT_PRESSURE_LEVELS
    # Список переменных ERA5, которые нужно скачать.
    variables: tuple[str, ...] = DEFAULT_VARIABLES


@dataclass(frozen=True, slots=True)
class WindConfig:
    # Путь к NetCDF-файлу с данными ветра.
    path: Path = DEFAULT_WIND_DATA_PATH
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
    # Полуширина окна случайного времени вокруг середины диапазона датасета.
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
    # Максимальное число шагов в одном эпизоде (защита от бесконечных эпизодов).
    max_episode_steps: int = MAX_EPISODE_STEPS


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
    # Каталог для чекпоинтов и метрик обучения.
    logdir: Path = Path("runs/ppo")
    # Количество параллельных сред. M1 Pro имеет 10 ядер (8P+2E),
    # больше 8 даёт только IPC-overhead без прироста скорости.
    n_envs: int = 8


@dataclass(frozen=True, slots=True)
class VisualizationConfig:
    # Размер окна визуализации.
    window_size: tuple[int, int] = WINDOW_SIZE
    # Нижний цвет фона сцены.
    bg_bottom: str = "deepskyblue"
    # Верхний цвет фона сцены.
    bg_top: str = "midnightblue"
    # Время старта визуализации.
    sim_start_time: np.datetime64 = SIM_TIME


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


