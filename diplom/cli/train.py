from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

from diplom.cli.training_options import (
    DATA_DIR_OPTION,
    DATASET_OPTION,
    DEVICE_OPTION,
    EXPERIMENT_OPTION,
    MANIFEST_OPTION,
    MODEL_OPTION,
    OBS_OPTION,
    RANDOMIZE_POSITION_OPTION,
    REWARD_OPTION,
    SEED_OPTION,
    START_TIME_OPTION,
    TARGET_RADIUS_OPTION,
    TIMESTEPS_OPTION,
    VERBOSE_OPTION,
    build_ppo_app_config,
    n_envs_option,
    ppo_training_options,
    training_logdir_option,
)


def train_ppo(
    total_timesteps: int = TIMESTEPS_OPTION,
    seed: int = SEED_OPTION,
    logdir: Path = training_logdir_option(),
    n_envs: int = n_envs_option(),
    device: str = DEVICE_OPTION,
    target_reach_radius: float = TARGET_RADIUS_OPTION,
    start_time: Optional[datetime] = START_TIME_OPTION,
    randomize_position: bool = RANDOMIZE_POSITION_OPTION,
    in_process: bool = typer.Option(
        False,
        "--in-process",
        help="Одна среда в DummyVecEnv (для profile-ppo-mem/cpu; не использовать в боевом обучении)",
    ),
    trajectories: bool = typer.Option(
        True,
        "--trajectories/--no-trajectories",
        help="HTML-траектории и JSONL шагов (отключите для максимальной скорости)",
    ),
    open_trajectories: bool = typer.Option(
        False,
        "--open-trajectories/--no-open-trajectories",
        help="Открыть HTML live-viewer траекторий в браузере при старте обучения (нужны --trajectories)",
    ),
    trajectory_wind_cones: bool = typer.Option(
        False,
        "--trajectory-wind-cones/--no-trajectory-wind-cones",
        help="Конусы ветра на HTML-графике траекторий (нужны --trajectories)",
    ),
    verbose: int = VERBOSE_OPTION,
    resume: bool = typer.Option(
        False,
        "--resume",
        help=(
            "Продолжить из {logdir}/{experiment|датасет}/ppo_model.zip: та же модель, тот же PPO_N, "
            "счётчик шагов и кривая TensorBoard без сброса"
        ),
    ),
    dataset: Optional[str] = DATASET_OPTION,
    data_dir: Path = DATA_DIR_OPTION,
    experiment: Optional[str] = EXPERIMENT_OPTION,
    manifest_path: Path = MANIFEST_OPTION,
    model: str = MODEL_OPTION,
    reward: str = REWARD_OPTION,
    obs: str = OBS_OPTION,
) -> None:
    """Запустить обучение PPO-модели."""
    from diplom.dev.profiling.runner import PROFILE_N_ENVS
    from diplom.rl.ppo.runner import train_ppo as run_train_ppo

    if open_trajectories and not trajectories:
        typer.echo(
            "[ошибка] --open-trajectories требует включённых --trajectories",
            err=True,
        )
        raise typer.Exit(code=1)

    opts = ppo_training_options(
        total_timesteps=total_timesteps,
        seed=seed,
        logdir=logdir,
        n_envs=PROFILE_N_ENVS if in_process else n_envs,
        device=device,
        verbose=verbose,
        target_reach_radius=target_reach_radius,
        start_time=start_time,
        randomize_position=randomize_position,
        dataset=dataset,
        data_dir=data_dir,
        experiment_name=experiment,
        manifest_path=manifest_path,
        model_name=model,
        reward_name=reward,
        obs_name=obs,
    )
    app_config = build_ppo_app_config(opts)
    if trajectory_wind_cones:
        app_config = replace(
            app_config,
            environment=replace(
                app_config.environment,
                trajectory_show_wind_cones=True,
            ),
        )
    try:
        run_train_ppo(
            app_config,
            force_dummy_vec_env=in_process,
            enable_trajectory_viz=trajectories,
            open_trajectory_viz=open_trajectories,
            resume=resume,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


def train_parallel_ppo(ctx: typer.Context) -> None:
    """Несколько train-ppo параллельно с одним процессом рендера траекторий.

    Из манифеста (``data/training/datasets_manifest.toml``):

    \b
      diplom train-parallel-ppo --from-manifest
      diplom train-parallel-ppo --from-manifest --jobs 2

    Вручную — глобально ``--jobs N``, затем блоки ``runner``:

    \b
      diplom train-parallel-ppo --jobs 2 runner --dataset era5_... --envs=2
    """
    from diplom.dev.parallel_ppo import run_train_parallel_ppo

    try:
        code = run_train_parallel_ppo(list(ctx.args))
    except ValueError as exc:
        typer.echo(f"[ошибка] {exc}", err=True)
        raise typer.Exit(code=1) from exc
    raise typer.Exit(code=code)
