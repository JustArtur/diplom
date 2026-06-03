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
    ERA5_TRAINING_MANIFEST_PATH,
    resolve_dataset_reference,
)
from diplom.config import DEFAULT_WINDOW_SIZE
from diplom.envs.constants import (
    TARGET_REACH_RADIUS,
    TRAIN_TARGET_POSITION_HORIZONTAL_DELTA,
    TRAIN_TARGET_POSITION_VERTICAL_DELTA,
)
from diplom.envs.rewards import get_reward_fn, list_reward_names
from diplom.envs.observations import get_obs_spec, list_obs_names
from diplom.rl.ppo.models import get_model_spec, list_model_specs

DEFAULT_TRAINING_LOGDIR = TrainingConfig().logdir
DEFAULT_PROFILE_LOGDIR = TrainingConfig().profile_logdir
DEFAULT_TIMESTEPS = TrainingConfig().total_timesteps
DEFAULT_SEED = TrainingConfig().seed
DEFAULT_N_ENVS = TrainingConfig().n_envs
DEFAULT_DEVICE = TrainingConfig().device
DEFAULT_VERBOSE = TrainingConfig().verbose
DEFAULT_TARGET_REACH_RADIUS = TARGET_REACH_RADIUS

_LOGDIR_HELP = (
    "Родительский каталог; run, {logdir}/{experiment|имя_датасета}/PPO_N "
    "(модель, TensorBoard, trajectories внутри run-каталога; "
    "имя датасета, NetCDF без .nc; см. --experiment)."
)
_START_TIME_HELP = (
    "Момент старта симуляции (ISO 8601). "
    "По умолчанию, первый шаг времени из датасета ERA5."
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
    help="Логирование PPO в консоль (SB3): 0, тихо, 1, таблица метрик",
)
TARGET_RADIUS_OPTION = typer.Option(
    DEFAULT_TARGET_REACH_RADIUS,
    "--target-radius",
    help="Радиус вокруг цели, при попадании в который эпизод считается успешным",
)
RANDOMIZE_INITIAL_POSITION_OPTION = typer.Option(
    False,
    "--randomize-initial-position/--no-randomize-initial-position",
    help="Случайное смещение стартовой позиции аэростата вокруг базовых координат",
)
RANDOMIZE_TARGET_POSITION_OPTION = typer.Option(
    False,
    "--randomize-target-position/--no-randomize-target-position",
    help=(
        "Случайное смещение целевой позиции вокруг базовых координат; "
        "если заданы нестандартные --randomize-target-horizontal-delta/--randomize-target-vertical-delta, "
        "флаг включится автоматически"
    ),
)
RANDOMIZE_TARGET_HORIZONTAL_DELTA_OPTION = typer.Option(
    TRAIN_TARGET_POSITION_HORIZONTAL_DELTA,
    "--randomize-target-horizontal-delta",
    min=0.0,
    help="Разброс целевой позиции по оси X (м)",
)
RANDOMIZE_TARGET_VERTICAL_DELTA_OPTION = typer.Option(
    TRAIN_TARGET_POSITION_VERTICAL_DELTA,
    "--randomize-target-vertical-delta",
    min=0.0,
    help="Разброс целевой позиции по вертикали (м)",
)
DATASET_OPTION = typer.Option(
    None,
    "--dataset",
    "-f",
    help="Имя, путь или id (#1) NetCDF ERA5 из datasets_manifest.toml",
)
EXPERIMENT_OPTION = typer.Option(
    "",
    "--experiment",
    "-x",
    help=(
        "Имя каталога run-а под --logdir; пусто, r-{reward}_o-{obs}_m-{model} "
        "(нужно то же имя для --resume)"
    ),
)
MANIFEST_OPTION = typer.Option(
    ERA5_TRAINING_MANIFEST_PATH,
    "--manifest",
    help="Манифест датасетов для разрешения --dataset по id",
)
DATA_DIR_OPTION = typer.Option(
    ERA5_TRAINING_DATA_DIR,
    "--data-dir",
    help="Каталог с датасетами для обучения (если --dataset задано как имя без пути)",
)
MODEL_OPTION = typer.Option(
    "default",
    "--model",
    help=f"PPO-политика из diplom.rl.ppo.models: {', '.join(list_model_specs())}",
)
REWARD_OPTION = typer.Option(
    "simple",
    "--reward",
    help=f"Reward-функция из diplom.envs.rewards: {', '.join(list_reward_names())}",
)
OBS_OPTION = typer.Option(
    "default",
    "--obs",
    help=f"Obs-модель из diplom.envs.observations: {', '.join(list_obs_names())}",
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
    randomize_initial_position: bool
    randomize_target_position: bool
    target_horizontal_delta: float
    target_vertical_delta: float
    dataset: str | None
    data_dir: Path
    experiment_name: str | None
    manifest_path: Path
    model_name: str
    reward_name: str
    obs_name: str


def ppo_experiment_name(
    *,
    reward_name: str,
    obs_name: str,
    model_name: str,
    experiment_name: str | None = None,
) -> str:
    if experiment_name and experiment_name.strip():
        return experiment_name.strip()
    return f"r-{reward_name}_o-{obs_name}_m-{model_name}"


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


def _should_randomize_target_position(options: PpoTrainingCliOptions) -> bool:
    return bool(
        options.randomize_target_position
        or options.target_horizontal_delta != TRAIN_TARGET_POSITION_HORIZONTAL_DELTA
        or options.target_vertical_delta != TRAIN_TARGET_POSITION_VERTICAL_DELTA
    )


def build_ppo_app_config(options: PpoTrainingCliOptions) -> AppConfig:
    get_model_spec(options.model_name)
    get_reward_fn(options.reward_name)
    get_obs_spec(options.obs_name)

    wind = WindConfig()
    if options.dataset is not None:
        wind = replace(
            wind,
            path=resolve_dataset_reference(
                options.dataset,
                data_dir=options.data_dir,
                manifest_path=options.manifest_path,
            ),
        )

    effective_logdir = options.logdir
    run_prefix = ppo_experiment_name(
        reward_name=options.reward_name,
        obs_name=options.obs_name,
        model_name=options.model_name,
        experiment_name=options.experiment_name,
    )

    config = AppConfig(
        wind=wind,
        environment=EnvironmentConfig(
            balloon=_balloon_config(options.start_time),
            target_reach_radius=options.target_reach_radius,
            randomize_initial_position=options.randomize_initial_position,
            randomize_target_position=_should_randomize_target_position(options),
            train_target_position_horizontal_delta=options.target_horizontal_delta,
            train_target_position_vertical_delta=options.target_vertical_delta,
            reward_name=options.reward_name,
            obs_name=options.obs_name,
        ),
        training=TrainingConfig(
            total_timesteps=options.total_timesteps,
            seed=options.seed,
            logdir=effective_logdir,
            n_envs=options.n_envs,
            device=options.device,
            verbose=options.verbose,
            experiment_name=run_prefix,
            model_name=options.model_name,
        ),
        visualization=_visualization_config(options.start_time),
    )

    return config


def balloon_config(start_time: datetime | None = None) -> BalloonConfig:
    return _balloon_config(start_time)


def build_default_app_config(*, start_time: datetime | None = None) -> AppConfig:
    # Минимальный AppConfig для viz/rollout без параметров обучения
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
            randomize_initial_position=False,
            randomize_target_position=False,
            target_horizontal_delta=TRAIN_TARGET_POSITION_HORIZONTAL_DELTA,
            target_vertical_delta=TRAIN_TARGET_POSITION_VERTICAL_DELTA,
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
    randomize_initial_position: bool,
    randomize_target_position: bool,
    target_horizontal_delta: float,
    target_vertical_delta: float,
    dataset: Optional[str],
    data_dir: Path,
    experiment_name: Optional[str] = None,
    manifest_path: Path = ERA5_TRAINING_MANIFEST_PATH,
    model_name: str = "default",
    reward_name: str = "simple",
    obs_name: str = "default",
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
        randomize_initial_position=randomize_initial_position,
        randomize_target_position=randomize_target_position,
        target_horizontal_delta=target_horizontal_delta,
        target_vertical_delta=target_vertical_delta,
        dataset=dataset,
        data_dir=data_dir,
        experiment_name=experiment_name.strip() if experiment_name else None,
        manifest_path=manifest_path,
        model_name=model_name,
        reward_name=reward_name,
        obs_name=obs_name,
    )
