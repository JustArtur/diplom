"""Общие CLI-опции и сборка AppConfig для train/profile PPO."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

from diplom.config import (
    AppConfig,
    BalloonConfig,
    EnvironmentConfig,
    TrainingConfig,
    VisualizationConfig,
    WindConfig,
)
from diplom.data.era5_paths import (
    ERA5_TRAINING_DATA_DIR,
    resolve_era5_dataset_path,
    training_logdir_for_dataset,
)
from diplom.config import DEFAULT_WINDOW_SIZE
from diplom.envs.constants import TARGET_REACH_RADIUS

DEFAULT_TRAINING_LOGDIR = TrainingConfig().logdir
DEFAULT_PROFILE_LOGDIR = TrainingConfig().profile_logdir
DEFAULT_TIMESTEPS = TrainingConfig().total_timesteps
DEFAULT_SEED = TrainingConfig().seed
DEFAULT_N_ENVS = TrainingConfig().n_envs
DEFAULT_DEVICE = TrainingConfig().device
DEFAULT_VERBOSE = TrainingConfig().verbose
DEFAULT_TARGET_REACH_RADIUS = TARGET_REACH_RADIUS

_LOGDIR_HELP = (
    "Родительский каталог; артефакты пишутся в {logdir}/{имя_датасета}/ "
    "(имя датасета — NetCDF без .nc)."
)
_START_TIME_HELP = (
    "Момент старта симуляции (ISO 8601). "
    "По умолчанию — первый шаг времени из датасета ERA5."
)

START_TIME_OPTION = typer.Option(None, "--start-time", help=_START_TIME_HELP)
TIMESTEPS_OPTION = typer.Option(
    DEFAULT_TIMESTEPS,
    "--timesteps",
    "-t",
    help="Количество шагов обучения",
)
SEED_OPTION = typer.Option(DEFAULT_SEED, "--seed", help="Seed для воспроизводимости")
DEVICE_OPTION = typer.Option(
    DEFAULT_DEVICE,
    "--device",
    "-d",
    help="Устройство для PPO: cpu, cuda или mps",
    case_sensitive=False,
)
VERBOSE_OPTION = typer.Option(
    DEFAULT_VERBOSE,
    "--verbose",
    "-v",
    help="Логирование PPO в консоль (SB3): 0 — тихо, 1 — таблица метрик",
)
TARGET_RADIUS_OPTION = typer.Option(
    DEFAULT_TARGET_REACH_RADIUS,
    "--target-radius",
    help="Радиус вокруг цели, при попадании в который эпизод считается успешным",
)
RANDOMIZE_POSITION_OPTION = typer.Option(
    True,
    "--randomize-position/--no-randomize-position",
    help="Случайное смещение стартовой позиции и цели вокруг базовых координат",
)
RANDOMIZE_TIME_OPTION = typer.Option(
    True,
    "--randomize-time/--no-randomize-time",
    help="Случайное время эпизода в окне вокруг середины диапазона датасета",
)
DATASET_OPTION = typer.Option(
    None,
    "--dataset",
    "-f",
    help="Имя или путь к NetCDF ERA5; по умолчанию — дефолтный датасет из конфига",
)
DATA_DIR_OPTION = typer.Option(
    ERA5_TRAINING_DATA_DIR,
    "--data-dir",
    help="Каталог с датасетами для обучения (если --dataset задано как имя без пути)",
)
PROFILE_MAIN_OPTION = typer.Option(
    False,
    "--profile-main",
    help="Профилировать главный процесс (PPO, callbacks)",
)
PROFILE_ENVS_OPTION = typer.Option(
    False,
    "--profile-envs",
    help="Профилировать воркеры SubprocVecEnv (env_000, env_001, …)",
)
PROFILE_TRAJECTORY_OPTION = typer.Option(
    False,
    "--profile-trajectory",
    help="Профилировать процесс рендера HTML траекторий",
)


def training_logdir_option(*, profile: bool = False) -> typer.Option:
    default = DEFAULT_PROFILE_LOGDIR if profile else DEFAULT_TRAINING_LOGDIR
    return typer.Option(default, "--logdir", "-l", help=_LOGDIR_HELP)


def n_envs_option(*, profile: bool = False) -> typer.Option:
    if profile:
        return typer.Option(
            DEFAULT_N_ENVS,
            "--envs",
            "-e",
            help="Число параллельных сред; по одному профилю на процесс env_NNN",
        )
    return typer.Option(
        DEFAULT_N_ENVS,
        "--envs",
        "-e",
        help="Количество параллельных сред",
    )


@dataclass(frozen=True, slots=True)
class PpoTrainingCliOptions:
    total_timesteps: int
    seed: int
    logdir: Path
    n_envs: int
    device: str
    verbose: int
    target_reach_radius: float
    start_time: datetime | None
    randomize_position: bool
    randomize_time: bool
    dataset: str | None
    data_dir: Path


def _balloon_config(start_time: datetime | None = None) -> BalloonConfig:
    import numpy as np

    balloon = BalloonConfig()
    if start_time is not None:
        return replace(balloon, sim_time=np.datetime64(start_time))
    return balloon


def _visualization_config(start_time: datetime | None = None) -> VisualizationConfig:
    import numpy as np

    viz = VisualizationConfig(
        window_size=DEFAULT_WINDOW_SIZE,
        bg_bottom="deepskyblue",
        bg_top="midnightblue",
    )
    if start_time is not None:
        return replace(viz, sim_start_time=np.datetime64(start_time))
    return viz


def build_ppo_app_config(options: PpoTrainingCliOptions) -> AppConfig:
    wind = WindConfig()
    if options.dataset is not None:
        wind = WindConfig(
            path=resolve_era5_dataset_path(options.dataset, data_dir=options.data_dir),
            origin_lat=wind.origin_lat,
            origin_lon=wind.origin_lon,
        )

    effective_logdir = training_logdir_for_dataset(wind.path, options.logdir)

    return AppConfig(
        wind=wind,
        environment=EnvironmentConfig(
            balloon=_balloon_config(options.start_time),
            target_reach_radius=options.target_reach_radius,
            randomize_start_state=options.randomize_position,
            randomize_start_time=options.randomize_time,
        ),
        training=TrainingConfig(
            total_timesteps=options.total_timesteps,
            seed=options.seed,
            logdir=effective_logdir,
            n_envs=options.n_envs,
            device=options.device,
            verbose=options.verbose,
        ),
        visualization=_visualization_config(options.start_time),
    )


def balloon_config(start_time: datetime | None = None) -> BalloonConfig:
    return _balloon_config(start_time)


def build_default_app_config(*, start_time: datetime | None = None) -> AppConfig:
    """Минимальный AppConfig для viz/rollout без параметров обучения."""
    return build_ppo_app_config(
        ppo_training_options(
            total_timesteps=DEFAULT_TIMESTEPS,
            seed=DEFAULT_SEED,
            logdir=DEFAULT_TRAINING_LOGDIR,
            n_envs=DEFAULT_N_ENVS,
            device=DEFAULT_DEVICE,
            verbose=DEFAULT_VERBOSE,
            target_reach_radius=DEFAULT_TARGET_REACH_RADIUS,
            start_time=start_time,
            randomize_position=False,
            randomize_time=False,
            dataset=None,
            data_dir=ERA5_TRAINING_DATA_DIR,
        )
    )


def ppo_training_options(
    *,
    total_timesteps: int,
    seed: int,
    logdir: Path,
    n_envs: int,
    device: str,
    verbose: int,
    target_reach_radius: float,
    start_time: Optional[datetime],
    randomize_position: bool,
    randomize_time: bool,
    dataset: Optional[str],
    data_dir: Path,
) -> PpoTrainingCliOptions:
    return PpoTrainingCliOptions(
        total_timesteps=total_timesteps,
        seed=seed,
        logdir=logdir,
        n_envs=n_envs,
        device=device,
        verbose=verbose,
        target_reach_radius=target_reach_radius,
        start_time=start_time,
        randomize_position=randomize_position,
        randomize_time=randomize_time,
        dataset=dataset,
        data_dir=data_dir,
    )
