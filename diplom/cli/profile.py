from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

from diplom.cli.training_options import (
    DATA_DIR_OPTION,
    DATASET_OPTION,
    DEVICE_OPTION,
    PROFILE_ENVS_OPTION,
    PROFILE_MAIN_OPTION,
    PROFILE_TRAJECTORY_OPTION,
    RANDOMIZE_POSITION_OPTION,
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


def profile_ppo_mem(
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Файл memray (.bin); по умолчанию <logdir>/PPO_N/memray.bin",
    ),
    flamegraph: Optional[Path] = typer.Option(
        None,
        "--flamegraph",
        help="HTML flame graph; по умолчанию <logdir>/PPO_N/memray.html",
    ),
    no_flamegraph: bool = typer.Option(
        False,
        "--no-flamegraph",
        help="Не строить HTML, только .bin и таблицу в терминале",
    ),
    no_table: bool = typer.Option(
        False,
        "--no-table",
        help="Не выводить memray table в терминал",
    ),
    native: bool = typer.Option(
        False,
        "--native",
        help="native_traces: стек C-расширений (PyTorch, NumPy); профиль тяжелее",
    ),
    total_timesteps: int = TIMESTEPS_OPTION,
    seed: int = SEED_OPTION,
    logdir: Path = training_logdir_option(profile=True),
    device: str = DEVICE_OPTION,
    verbose: int = VERBOSE_OPTION,
    target_reach_radius: float = TARGET_RADIUS_OPTION,
    start_time: Optional[datetime] = START_TIME_OPTION,
    randomize_position: bool = RANDOMIZE_POSITION_OPTION,
    n_envs: int = n_envs_option(profile=True),
    single_process: bool = typer.Option(
        False,
        "--single-process",
        help="Одна среда в DummyVecEnv, один memray-файл (как раньше; для быстрой отладки)",
    ),
    profile_main: bool = PROFILE_MAIN_OPTION,
    profile_envs: bool = PROFILE_ENVS_OPTION,
    profile_trajectory: bool = PROFILE_TRAJECTORY_OPTION,
    dataset: Optional[str] = DATASET_OPTION,
    data_dir: Path = DATA_DIR_OPTION,
) -> None:
    """Профиль памяти при обучении PPO (memray).

    Профилирование выключено, пока не передан хотя бы один флаг --profile-*.
    Каждый включённый процесс пишет свой файл в <logdir>/PPO_N/memray/<имя>.bin.

    CPU (время): diplom profile-ppo-cpu. Запускайте отдельно — совмещение сильно замедляет прогон.

    Установка: poetry install --with dev

    \b
      diplom profile-ppo-mem -t 50000 -e 8 -f era5_... --profile-main --profile-envs
      diplom profile-ppo-mem -t 50000 --profile-main
      diplom profile-ppo-mem -t 50000 --single-process --profile-main
      open profile_ppo/{датасет}/PPO_0/memray/main.html
    """
    from diplom.dev.profiling.memory import MemrayProfileTargets
    from diplom.dev.profiling.runner import PROFILE_N_ENVS, MemrayNotFoundError, run_memray_train

    profile_targets = MemrayProfileTargets(
        main=profile_main,
        envs=profile_envs,
        trajectory=profile_trajectory,
    )
    effective_n_envs = PROFILE_N_ENVS if single_process else n_envs
    app_config = build_ppo_app_config(
        ppo_training_options(
            total_timesteps=total_timesteps,
            seed=seed,
            logdir=logdir,
            n_envs=effective_n_envs,
            device=device,
            verbose=verbose,
            target_reach_radius=target_reach_radius,
            start_time=start_time,
            randomize_position=randomize_position,
            dataset=dataset,
            data_dir=data_dir,
        )
    )
    try:
        run_dir, reports = run_memray_train(
            app_config,
            output=output,
            flamegraph=flamegraph,
            skip_flamegraph=no_flamegraph,
            native_traces=native,
            print_table=not no_table,
            multiprocess=not single_process,
            profile_targets=profile_targets,
        )
    except MemrayNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Run: {run_dir}")
    for report in reports:
        typer.echo(f"  {report.process_name}: {report.bin_path}")
        if report.html_path is not None:
            typer.echo(f"    flame graph: {report.html_path}")


def profile_ppo_cpu(
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Файл cProfile; по умолчанию <logdir>/PPO_N/profile.prof",
    ),
    top_lines: int = typer.Option(40, "--top", help="Сколько строк вывести в таблице"),
    total_timesteps: int = TIMESTEPS_OPTION,
    seed: int = SEED_OPTION,
    logdir: Path = training_logdir_option(profile=True),
    device: str = DEVICE_OPTION,
    verbose: int = VERBOSE_OPTION,
    target_reach_radius: float = TARGET_RADIUS_OPTION,
    start_time: Optional[datetime] = START_TIME_OPTION,
    randomize_position: bool = RANDOMIZE_POSITION_OPTION,
    n_envs: int = n_envs_option(profile=True),
    single_process: bool = typer.Option(
        False,
        "--single-process",
        help="Одна среда в DummyVecEnv, один .prof (для быстрой отладки)",
    ),
    profile_main: bool = PROFILE_MAIN_OPTION,
    profile_envs: bool = PROFILE_ENVS_OPTION,
    profile_trajectory: bool = PROFILE_TRAJECTORY_OPTION,
    no_stats: bool = typer.Option(
        False,
        "--no-stats",
        help="Не выводить таблицу pstats в терминал",
    ),
    dataset: Optional[str] = DATASET_OPTION,
    data_dir: Path = DATA_DIR_OPTION,
) -> None:
    """Профиль CPU (cProfile) при обучении PPO.

    Профилирование выключено, пока не передан хотя бы один флаг --profile-*.
    Каждый включённый процесс пишет свой файл в <logdir>/PPO_N/cprofile/<имя>.prof.

    Память: diplom profile-ppo-mem (memray). Запускайте отдельно от profile-ppo-cpu.

    \b
      diplom profile-ppo-cpu -t 50000 -e 8 -f era5_... --profile-main --profile-envs
      diplom profile-ppo-cpu -t 50000 --single-process --profile-main
      snakeviz profile_ppo/{датасет}/PPO_0/cprofile/main.prof
    """
    from diplom.dev.profiling.memory import MemrayProfileTargets
    from diplom.dev.profiling.runner import PROFILE_N_ENVS, run_cprofile_train

    profile_targets = MemrayProfileTargets(
        main=profile_main,
        envs=profile_envs,
        trajectory=profile_trajectory,
    )
    effective_n_envs = PROFILE_N_ENVS if single_process else n_envs
    app_config = build_ppo_app_config(
        ppo_training_options(
            total_timesteps=total_timesteps,
            seed=seed,
            logdir=logdir,
            n_envs=effective_n_envs,
            device=device,
            verbose=verbose,
            target_reach_radius=target_reach_radius,
            start_time=start_time,
            randomize_position=randomize_position,
            dataset=dataset,
            data_dir=data_dir,
        )
    )
    try:
        run_dir, reports = run_cprofile_train(
            app_config,
            output=output,
            top_lines=top_lines,
            multiprocess=not single_process,
            profile_targets=profile_targets,
            print_stats=not no_stats,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Run: {run_dir}")
    for report in reports:
        typer.echo(f"  {report.process_name}: {report.prof_path}")
