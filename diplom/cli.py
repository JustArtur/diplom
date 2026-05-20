"""CLI-интерфейс для симулятора стратостата и RL-обучения."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Optional
import webbrowser

import numpy as np
import typer
from dotenv import load_dotenv

from diplom.config import (
    AppConfig,
    BalloonConfig,
    DownloadConfig,
    EnvironmentConfig,
    TrainingConfig,
    VisualizationConfig,
    WindConfig,
)
from diplom.data.era5_paths import (
    DEFAULT_ERA5_DATA_DIR,
    era5_outfile_for_bounds,
    list_era5_datasets,
    resolve_era5_dataset_path,
    wind_plot_html_path,
)

# Загружаем переменные окружения (.env) — ключи CDS API и т.п.
load_dotenv()

# Главный объект Typer CLI
app = typer.Typer(help="CLI утилиты для симулятора стратостата и RL.")

# Дефолты для CLI держим отдельно, чтобы команда читалась как слой сборки, а не как источник правил.
DEFAULT_DOWNLOAD_CONFIG = DownloadConfig()
DEFAULT_ENVIRONMENT_CONFIG = EnvironmentConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_VISUALIZATION_CONFIG = VisualizationConfig()
DEFAULT_ROLLOUT_MODEL_PATH = DEFAULT_TRAINING_CONFIG.logdir / "ppo_model.zip"

_RUN_NAME_HELP = (
    "Имя run-а; каталог будет {имя}#PPO_N (индекс как обычно по порядку). "
    "Без флага — PPO_N"
)

_START_TIME_HELP = (
    "Момент старта симуляции (ISO 8601). "
    "По умолчанию — первый шаг времени из датасета ERA5."
)
START_TIME_OPTION = typer.Option(None, "--start-time", help=_START_TIME_HELP)


def _balloon_config(
    start_time: datetime | None = None,
    **kwargs: object,
) -> BalloonConfig:
    balloon = BalloonConfig(**kwargs)
    if start_time is not None:
        balloon = replace(balloon, sim_time=np.datetime64(start_time))
    return balloon


def _visualization_config(start_time: datetime | None = None) -> VisualizationConfig:
    viz = VisualizationConfig(
        window_size=DEFAULT_VISUALIZATION_CONFIG.window_size,
        bg_bottom=DEFAULT_VISUALIZATION_CONFIG.bg_bottom,
        bg_top=DEFAULT_VISUALIZATION_CONFIG.bg_top,
    )
    if start_time is not None:
        viz = replace(viz, sim_start_time=np.datetime64(start_time))
    return viz


def _build_app_config(
    *,
    total_timesteps: int = DEFAULT_TRAINING_CONFIG.total_timesteps,
    seed: int = DEFAULT_TRAINING_CONFIG.seed,
    logdir: Path = DEFAULT_TRAINING_CONFIG.logdir,
    n_envs: int = DEFAULT_TRAINING_CONFIG.n_envs,
    device: str = DEFAULT_TRAINING_CONFIG.device,
    verbose: int = DEFAULT_TRAINING_CONFIG.verbose,
    use_worker_policy_rollout: bool = DEFAULT_TRAINING_CONFIG.use_worker_policy_rollout,
    target_reach_radius: float = DEFAULT_ENVIRONMENT_CONFIG.target_reach_radius,
    start_time: datetime | None = None,
    randomize_start_state: bool = True,
    randomize_start_time: bool = True,
    dataset: str | None = None,
    data_dir: Path = DEFAULT_ERA5_DATA_DIR,
) -> AppConfig:
    wind = WindConfig()
    if dataset is not None:
        wind = replace(wind, path=resolve_era5_dataset_path(dataset, data_dir=data_dir))

    return AppConfig(
        wind=wind,
        environment=EnvironmentConfig(
            balloon=_balloon_config(start_time),
            target_reach_radius=target_reach_radius,
            randomize_start_state=randomize_start_state,
            randomize_start_time=randomize_start_time,
        ),
        training=TrainingConfig(
            total_timesteps=total_timesteps,
            seed=seed,
            logdir=logdir,
            n_envs=n_envs,
            device=device,
            verbose=verbose,
            use_worker_policy_rollout=use_worker_policy_rollout,
        ),
        visualization=_visualization_config(start_time),
    )


# ──────────────────── download ────────────────────

@app.command()
def download(
    outfile: Path | None = typer.Option(
        None,
        "--outfile",
        "-o",
        help="Путь к итоговому NetCDF; по умолчанию era5_{north}_{south}_{west}_{east}_{start}_{end}.nc",
    ),
    north: float = typer.Option(DEFAULT_DOWNLOAD_CONFIG.north, help="Северная граница широты"),
    west: float = typer.Option(DEFAULT_DOWNLOAD_CONFIG.west, help="Западная граница долготы"),
    south: float = typer.Option(DEFAULT_DOWNLOAD_CONFIG.south, help="Южная граница широты"),
    east: float = typer.Option(DEFAULT_DOWNLOAD_CONFIG.east, help="Восточная граница долготы"),
    start: str = typer.Option(DEFAULT_DOWNLOAD_CONFIG.start, help="Начало периода YYYY-MM-DD"),
    end: str = typer.Option(DEFAULT_DOWNLOAD_CONFIG.end, help="Конец периода YYYY-MM-DD"),
    level: list[str] = typer.Option(
        list(DEFAULT_DOWNLOAD_CONFIG.pressure_levels),
        "--level", "-l",
        help="Уровни давления hPa; можно повторять.",
        show_default=False,
    ),
    variable: list[str] = typer.Option(
        list(DEFAULT_DOWNLOAD_CONFIG.variables),
        "--var", "-v",
        help="Имена переменных CDS; можно повторять.",
        show_default=False,
    ),
    chunks_dir: Path | None = typer.Option(
        None,
        "--chunks-dir",
        help="Каталог для дневных чанков; по умолчанию {outfile.stem}.chunks рядом с outfile.",
    ),
    keep_chunks: bool = typer.Option(
        False,
        "--keep-chunks",
        help="Не удалять дневные NetCDF после склейки.",
    ),
    workers: int | None = typer.Option(
        None,
        "--workers", "-j",
        min=1,
        help="Параллельная загрузка по дням с последующей склейкой (2–4 обычно безопасно). "
        "Без флага — один запрос CDS сразу в outfile.",
    ),
    hour_step: int = typer.Option(
        DEFAULT_DOWNLOAD_CONFIG.hour_step,
        "--hour-step",
        min=1,
        max=24,
        help="Шаг по часам в запросе CDS: 2 → 00:00, 02:00, …, 22:00 (12 точек в сутки).",
    ),
) -> None:
    """Скачать подмножество ERA5 в NetCDF."""
    from diplom.data.era5_download import download_era5_pressure

    resolved_outfile = outfile or era5_outfile_for_bounds(
        north=north,
        south=south,
        west=west,
        east=east,
        start=start,
        end=end,
    )

    download_era5_pressure(
        DownloadConfig(
            outfile=resolved_outfile,
            north=north,
            west=west,
            south=south,
            east=east,
            start=start,
            end=end,
            pressure_levels=tuple(level),
            variables=tuple(variable),
            hour_step=hour_step,
        ),
        chunks_dir=chunks_dir,
        keep_chunks=keep_chunks,
        workers=workers,
    )

# ──────────────────── viz_real ────────────────────

@app.command()
def viz_real(
    start_time: Optional[datetime] = START_TIME_OPTION,
) -> None:
    """Запуск PyVista-визуализации на реальном ветре."""

    from diplom.viz.visualization_runner import VisualizationRunner

    app_config = _build_app_config(start_time=start_time)
    VisualizationRunner().run_real(app_config)


# ──────────────────── train_ppo ────────────────────

@app.command("train-ppo")
def train_ppo(
    total_timesteps: int = typer.Option(
        DEFAULT_TRAINING_CONFIG.total_timesteps,
        "--timesteps",
        "-t",
        help="Количество шагов обучения",
    ),
    seed: int = typer.Option(DEFAULT_TRAINING_CONFIG.seed, "--seed", help="Seed для воспроизводимости"),
    logdir: Path = typer.Option(
        DEFAULT_TRAINING_CONFIG.logdir,
        "--logdir",
        "-l",
        help="Каталог для артефактов обучения",
    ),
    n_envs: int = typer.Option(DEFAULT_TRAINING_CONFIG.n_envs, "--envs", "-e", help="Количество параллельных сред"),
    device: str = typer.Option(
        DEFAULT_TRAINING_CONFIG.device,
        "--device", "-d",
        help="Устройство для нейросети PPO: cpu, cuda или mps",
        case_sensitive=False,
    ),
    target_reach_radius: float = typer.Option(
        DEFAULT_ENVIRONMENT_CONFIG.target_reach_radius,
        "--target-radius",
        help="Радиус вокруг цели, при попадании в который эпизод считается успешным",
    ),
    start_time: Optional[datetime] = START_TIME_OPTION,
    randomize_position: bool = typer.Option(
        True,
        "--randomize-position/--no-randomize-position",
        help="Случайное смещение стартовой позиции и цели вокруг базовых координат",
    ),
    randomize_time: bool = typer.Option(
        True,
        "--randomize-time/--no-randomize-time",
        help="Случайное время эпизода в окне вокруг середины диапазона датасета",
    ),
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
    main_policy_rollout: bool = typer.Option(
        False,
        "--main-policy-rollout",
        help="Отладка: policy+step в main (ShmemSubprocVecEnv), без гибрида worker+shmem",
    ),
    verbose: int = typer.Option(
        DEFAULT_TRAINING_CONFIG.verbose,
        "--verbose",
        "-v",
        help="Уровень логирования PPO в консоль (SB3): 0 — тихо, 1 — таблица метрик; env-метрики только в TensorBoard",
    ),
    run_name: Optional[str] = typer.Option(
        None,
        "--run-name",
        help=_RUN_NAME_HELP,
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Продолжить из runs/ppo/ppo_model.zip (без флага — новая модель)",
    ),
    continue_tensorboard: bool = typer.Option(
        False,
        "--continue-tensorboard/--new-tensorboard-run",
        help=(
            "При --resume дописывать TensorBoard в последний PPO_N "
            "(с учётом --run-name; иначе — новый каталог run-а)"
        ),
    ),
    dataset: Optional[str] = typer.Option(
        None,
        "--dataset",
        "-f",
        help="Имя или путь к NetCDF ERA5; по умолчанию — дефолтный датасет из конфига",
    ),
    data_dir: Path = typer.Option(
        DEFAULT_ERA5_DATA_DIR,
        "--data-dir",
        help="Каталог с датасетами (если --dataset задано как имя без пути)",
    ),
) -> None:
    """Запустить обучение PPO-модели."""
    from diplom.train.ppo_runner import train_ppo as run_train_ppo
    from diplom.train.profiling import PROFILE_N_ENVS

    # Верхний слой задаёт только пользовательские параметры обучения.
    app_config = _build_app_config(
        total_timesteps=total_timesteps,
        seed=seed,
        logdir=logdir,
        n_envs=PROFILE_N_ENVS if in_process else n_envs,
        device=device,
        verbose=verbose,
        use_worker_policy_rollout=not main_policy_rollout,
        target_reach_radius=target_reach_radius,
        start_time=start_time,
        randomize_start_state=randomize_position,
        randomize_start_time=randomize_time,
        dataset=dataset,
        data_dir=data_dir,
    )
    try:
        run_train_ppo(
            app_config,
            force_dummy_vec_env=in_process,
            enable_trajectory_viz=trajectories,
            run_name=run_name,
            resume=resume,
            continue_tensorboard=continue_tensorboard,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


# ──────────────────── export_tensorboard ────────────────────

@app.command("export-tensorboard")
def export_tensorboard(
    path: Path = typer.Argument(
        ...,
        help="Файл events.out.tfevents.* или каталог (tb_1, PPO_N, runs/ppo)",
    ),
    recursive: bool = typer.Option(
        True,
        "--recursive/--no-recursive",
        help="Искать event-файлы в подкаталогах",
    ),
) -> None:
    """Экспорт scalar-метрик TensorBoard в CSV рядом с каждым event-файлом.

    Создаёт файлы вида ``events.out.tfevents.<id>.scalars.csv`` с колонками:
    tag, step, value, wall_time.

    \b
    Примеры:

      diplom export-tensorboard runs/ppo/PPO_25/tb_1

      diplom export-tensorboard runs/ppo/PPO_25/tb_1/events.out.tfevents.1779218441.host.0
    """
    from diplom.train.tensorboard_export import export_tensorboard_path

    try:
        results = export_tensorboard_path(path, recursive=recursive)
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    for item in results:
        typer.echo(f"{item.output}  ({item.rows} строк, из {item.source.name})")


# ──────────────────── profile_ppo_mem / profile_ppo_cpu ────────────────────

@app.command("profile-ppo-mem")
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
    total_timesteps: int = typer.Option(
        DEFAULT_TRAINING_CONFIG.total_timesteps,
        "--timesteps",
        "-t",
        help="Количество шагов обучения (как у train-ppo)",
    ),
    seed: int = typer.Option(DEFAULT_TRAINING_CONFIG.seed, "--seed", help="Seed"),
    logdir: Path = typer.Option(
        DEFAULT_TRAINING_CONFIG.profile_logdir,
        "--logdir",
        "-l",
        help="Каталог run-ов обучения (PPO_N/tb, PPO_N/trajectories)",
    ),
    device: str = typer.Option(
        DEFAULT_TRAINING_CONFIG.device,
        "--device",
        "-d",
        help="Устройство PPO: cpu, cuda, mps",
        case_sensitive=False,
    ),
    verbose: int = typer.Option(
        DEFAULT_TRAINING_CONFIG.verbose,
        "--verbose",
        "-v",
        help="Логирование PPO в консоль (SB3): 0 — тихо, 1 — таблица метрик",
    ),
    target_reach_radius: float = typer.Option(
        DEFAULT_ENVIRONMENT_CONFIG.target_reach_radius,
        "--target-radius",
        help="Радиус успешного завершения эпизода",
    ),
    start_time: Optional[datetime] = START_TIME_OPTION,
    randomize_position: bool = typer.Option(
        True,
        "--randomize-position/--no-randomize-position",
        help="Случайное смещение стартовой позиции и цели вокруг базовых координат",
    ),
    randomize_time: bool = typer.Option(
        True,
        "--randomize-time/--no-randomize-time",
        help="Случайное время эпизода в окне вокруг середины диапазона датасета",
    ),
    n_envs: int = typer.Option(
        DEFAULT_TRAINING_CONFIG.n_envs,
        "--envs",
        "-e",
        help="Число параллельных сред (SubprocVecEnv); по одному memray на процесс env_NNN",
    ),
    single_process: bool = typer.Option(
        False,
        "--single-process",
        help="Одна среда в DummyVecEnv, один memray-файл (как раньше; для быстрой отладки)",
    ),
    profile_main: bool = typer.Option(
        False,
        "--profile-main",
        help="Профилировать главный процесс (PPO, callbacks)",
    ),
    profile_envs: bool = typer.Option(
        False,
        "--profile-envs",
        help="Профилировать воркеры SubprocVecEnv (env_000, env_001, …)",
    ),
    profile_trajectory: bool = typer.Option(
        False,
        "--profile-trajectory",
        help="Профилировать процесс рендера HTML траекторий",
    ),
    main_policy_rollout: bool = typer.Option(
        False,
        "--main-policy-rollout",
        help="Policy+step в main (без гибрида worker+shmem)",
    ),
    run_name: Optional[str] = typer.Option(
        None,
        "--run-name",
        help=_RUN_NAME_HELP,
    ),
) -> None:
    """Профиль памяти при обучении PPO (memray).

    Профилирование выключено, пока не передан хотя бы один флаг --profile-*.
    Каждый включённый процесс пишет свой файл в <logdir>/PPO_N/memray/<имя>.bin.

    CPU (время): diplom profile-ppo-cpu. Запускайте отдельно — совмещение сильно замедляет прогон.

    Установка: poetry install --with dev

    \b
      diplom profile-ppo-mem -t 50000 -e 8 --profile-main --profile-envs --profile-trajectory
      diplom profile-ppo-mem -t 50000 --profile-main
      diplom profile-ppo-mem -t 50000 --single-process --profile-main
      open runs/profile_ppo/PPO_0/memray/main.html
    """
    from diplom.train.memory_profiling import MemrayProfileTargets
    from diplom.train.profiling import PROFILE_N_ENVS, MemrayNotFoundError, run_memray_train

    profile_targets = MemrayProfileTargets(
        main=profile_main,
        envs=profile_envs,
        trajectory=profile_trajectory,
    )
    effective_n_envs = PROFILE_N_ENVS if single_process else n_envs
    app_config = _build_app_config(
        total_timesteps=total_timesteps,
        seed=seed,
        logdir=logdir,
        n_envs=effective_n_envs,
        device=device,
        verbose=verbose,
        use_worker_policy_rollout=not main_policy_rollout,
        target_reach_radius=target_reach_radius,
        start_time=start_time,
        randomize_start_state=randomize_position,
        randomize_start_time=randomize_time,
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
            run_name=run_name,
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


@app.command("profile-ppo-cpu")
def profile_ppo_cpu(
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Файл cProfile; по умолчанию <logdir>/PPO_N/profile.prof",
    ),
    top_lines: int = typer.Option(40, "--top", help="Сколько строк вывести в таблице"),
    total_timesteps: int = typer.Option(
        DEFAULT_TRAINING_CONFIG.total_timesteps,
        "--timesteps",
        "-t",
    ),
    seed: int = typer.Option(DEFAULT_TRAINING_CONFIG.seed, "--seed"),
    logdir: Path = typer.Option(
        DEFAULT_TRAINING_CONFIG.profile_logdir,
        "--logdir",
        "-l",
        help="Каталог run-ов обучения (PPO_N/tb, PPO_N/trajectories)",
    ),
    device: str = typer.Option(
        DEFAULT_TRAINING_CONFIG.device,
        "--device",
        "-d",
        case_sensitive=False,
    ),
    verbose: int = typer.Option(
        DEFAULT_TRAINING_CONFIG.verbose,
        "--verbose",
        "-v",
        help="Логирование PPO в консоль (SB3): 0 — тихо, 1 — таблица метрик",
    ),
    target_reach_radius: float = typer.Option(
        DEFAULT_ENVIRONMENT_CONFIG.target_reach_radius,
        "--target-radius",
    ),
    start_time: Optional[datetime] = START_TIME_OPTION,
    randomize_position: bool = typer.Option(
        True,
        "--randomize-position/--no-randomize-position",
        help="Случайное смещение стартовой позиции и цели вокруг базовых координат",
    ),
    randomize_time: bool = typer.Option(
        True,
        "--randomize-time/--no-randomize-time",
        help="Случайное время эпизода в окне вокруг середины диапазона датасета",
    ),
    n_envs: int = typer.Option(
        DEFAULT_TRAINING_CONFIG.n_envs,
        "--envs",
        "-e",
        help="Число параллельных сред (SubprocVecEnv); по одному .prof на процесс env_NNN",
    ),
    single_process: bool = typer.Option(
        False,
        "--single-process",
        help="Одна среда в DummyVecEnv, один .prof (для быстрой отладки)",
    ),
    profile_main: bool = typer.Option(
        False,
        "--profile-main",
        help="Профилировать главный процесс (PPO, callbacks)",
    ),
    profile_envs: bool = typer.Option(
        False,
        "--profile-envs",
        help="Профилировать воркеры SubprocVecEnv (env_000, env_001, …)",
    ),
    profile_trajectory: bool = typer.Option(
        False,
        "--profile-trajectory",
        help="Профилировать процесс рендера HTML траекторий",
    ),
    no_stats: bool = typer.Option(
        False,
        "--no-stats",
        help="Не выводить таблицу pstats в терминал",
    ),
    main_policy_rollout: bool = typer.Option(
        False,
        "--main-policy-rollout",
        help="Policy+step в main (без гибрида worker+shmem)",
    ),
    run_name: Optional[str] = typer.Option(
        None,
        "--run-name",
        help=_RUN_NAME_HELP,
    ),
) -> None:
    """Профиль CPU (cProfile) при обучении PPO.

    Профилирование выключено, пока не передан хотя бы один флаг --profile-*.
    Каждый включённый процесс пишет свой файл в <logdir>/PPO_N/cprofile/<имя>.prof.

    Память: diplom profile-ppo-mem (memray). Запускайте отдельно от profile-ppo-cpu.

    \b
      diplom profile-ppo-cpu -t 50000 -e 8 --profile-main --profile-envs
      diplom profile-ppo-cpu -t 50000 --single-process --profile-main
      snakeviz runs/profile_ppo/PPO_0/cprofile/main.prof
    """
    from diplom.train.memory_profiling import MemrayProfileTargets
    from diplom.train.profiling import PROFILE_N_ENVS, run_cprofile_train

    profile_targets = MemrayProfileTargets(
        main=profile_main,
        envs=profile_envs,
        trajectory=profile_trajectory,
    )
    effective_n_envs = PROFILE_N_ENVS if single_process else n_envs
    app_config = _build_app_config(
        total_timesteps=total_timesteps,
        seed=seed,
        logdir=logdir,
        n_envs=effective_n_envs,
        device=device,
        verbose=verbose,
        use_worker_policy_rollout=not main_policy_rollout,
        target_reach_radius=target_reach_radius,
        start_time=start_time,
        randomize_start_state=randomize_position,
        randomize_start_time=randomize_time,
    )
    try:
        run_dir, reports = run_cprofile_train(
            app_config,
            output=output,
            top_lines=top_lines,
            multiprocess=not single_process,
            profile_targets=profile_targets,
            print_stats=not no_stats,
            run_name=run_name,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Run: {run_dir}")
    for report in reports:
        typer.echo(f"  {report.process_name}: {report.prof_path}")


# ──────────────────── rollout ────────────────────

@app.command("rollout")
def rollout(
    model_path: Path = typer.Option(
        DEFAULT_ROLLOUT_MODEL_PATH,
        "--model-path",
        "-m",
        help="Путь к сохранённой PPO-модели",
    ),
    episodes: int = typer.Option(1, "--episodes", "-n", help="Количество эпизодов"),
    seed: int = typer.Option(DEFAULT_TRAINING_CONFIG.seed, "--seed", help="Seed для воспроизводимости"),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Куда сохранить результаты rollout в JSON",
    ),
    render: bool = typer.Option(
        False,
        "--render/--no-render",
        help="Печатать состояние среды на каждом шаге",
    ),
    target_reach_radius: float = typer.Option(
        DEFAULT_ENVIRONMENT_CONFIG.target_reach_radius,
        "--target-radius",
        help="Радиус вокруг цели, при попадании в который эпизод считается успешным",
    ),
    plot_output: Path | None = typer.Option(
        None,
        "--plot",
        "-p",
        help="Путь для сохранения интерактивного 3D-графика траекторий (HTML). "
             "Если не указан — график не строится.",
    ),
    start_time: Optional[datetime] = START_TIME_OPTION,
    device: str = typer.Option(
        DEFAULT_TRAINING_CONFIG.device,
        "--device", "-d",
        help="Устройство для загрузки PPO: cpu, cuda или mps",
        case_sensitive=False,
    ),
) -> None:
    """Запустить rollout обученной модели и собрать траектории эпизодов."""
    from diplom.sim.rollout import rollout_episodes

    app_config = AppConfig(
        environment=EnvironmentConfig(
            balloon=_balloon_config(start_time),
            target_reach_radius=target_reach_radius,
        ),
        training=replace(DEFAULT_TRAINING_CONFIG, device=device),
    )
    try:
        results = rollout_episodes(
            app_config,
            n_episodes=episodes,
            policy_path=str(model_path),
            render=render,
            seed=seed,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    serialized_results = [asdict(result) for result in results]
    summary = {
        "episodes": episodes,
        "seed": seed,
        "model_path": str(model_path),
        "results": serialized_results,
    }

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    for idx, result in enumerate(results, start=1):
        typer.echo(
            f"episode={idx} success={result.success} steps={result.steps} total_reward={result.total_reward:.3f}"
        )

    if output is not None:
        typer.echo(f"saved={output}")

    if plot_output is not None:
        from diplom.viz.trajectory_plot import (
            EpisodeVizData,
            build_figure,
            compute_trajectory_bounds,
            save_figure,
        )
        from diplom.wind.factory import build_wind_interpolator

        viz_episodes = [
            EpisodeVizData(
                env_idx=idx,
                steps=result.trajectory,
                target_position=np.array(result.target_position, dtype=np.float32),
                label=(
                    f"episode {idx + 1} "
                    f"({'успех' if result.success else 'truncated'}, "
                    f"{result.steps} шагов)"
                ),
            )
            for idx, result in enumerate(results)
        ]
        wind_interpolator = build_wind_interpolator(app_config.wind)
        try:
            bounds = compute_trajectory_bounds(viz_episodes, world_bounds=wind_interpolator.world_bounds)
            fig = build_figure(
                episodes=viz_episodes,
                title=f"Rollout · {model_path.name} · {episodes} эпизодов",
                bounds=bounds,
            )
        finally:
            wind_interpolator.close()
        save_figure(fig, plot_output)
        typer.echo(f"plot saved={plot_output}")


# ──────────────────── wind_viz ────────────────────

@app.command("wind-viz")
def wind_viz(
    wind_file: Optional[Path] = typer.Option(
        None,
        "--wind-file", "-f",
        help="Один ERA5 NetCDF; без флага — все *.nc из --data-dir",
    ),
    data_dir: Path = typer.Option(
        DEFAULT_ERA5_DATA_DIR,
        "--data-dir",
        help="Каталог с ERA5 NetCDF (обрабатывается, если --wind-file не задан)",
    ),
    time: Optional[datetime] = typer.Option(
        None,
        "--time", "-t",
        help=(
            "Временна́я метка среза ERA5 (ISO 8601, например 2024-07-01T12:00:00). "
            "Если не задано — используется первый временной шаг датасета."
        ),
        formats=[
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M",
            "%Y-%m-%d",
        ],
    ),
    output: Path = typer.Option(
        Path("runs/wind"),
        "--output", "-o",
        help="Каталог для HTML-графиков (имя файла = имя датасета без .nc)",
    ),
    stride_lon: int = typer.Option(
        1, "--stride-lon",
        help="Прореживание по долготе (1 = каждая точка, 2 = через одну, ...)",
    ),
    stride_lat: int = typer.Option(
        1, "--stride-lat",
        help="Прореживание по широте",
    ),
    stride_altitude_m: float = typer.Option(
        500.0,
        "--stride-altitude-m",
        help="Шаг по высоте между конусами, м (ветер интерполируется по вертикали)",
    ),
    w_scale: float = typer.Option(
        0.0, "--w-scale",
        help="Масштаб вертикальной компоненты w для наглядности стрелок",
    ),
    open_browser: bool = typer.Option(
        True, "--open/--no-open",
        help="Открыть результат в браузере после сохранения",
    ),
    list_times: bool = typer.Option(
        False, "--list-times",
        help="Вывести все доступные временны́е метки в датасете и выйти",
    ),
    workers: int | None = typer.Option(
        None,
        "--workers", "-j",
        min=1,
        help="Число процессов для параллельной отрисовки. "
        "По умолчанию: min(число новых графиков, число CPU).",
    ),
) -> None:
    """Построить интерактивные 3D-графы поля ветра ERA5.

    По умолчанию обходит все ``*.nc`` в ``data/`` и сохраняет HTML в ``runs/wind/``.
    Заголовок графика совпадает с именем датасета; уже существующие файлы пропускаются.

    Примеры:

    \b
      # Все датасеты из data/
      diplom wind-viz

    \b
      # Список доступных временных меток (все датасеты или один файл)
      diplom wind-viz --list-times

    \b
      # Параллельно 4 процесса
      diplom wind-viz -j 4 --stride-lat 2 --stride-lon 2

    \b
      # Один файл, конкретное время
      diplom wind-viz -f data/era5_....nc --time 2024-07-01T12:00:00 --stride-lat 2
    """
    from diplom.viz.wind_plot import (
        WindPlotRenderJob,
        list_available_times,
        render_wind_plots,
    )

    if wind_file is not None:
        dataset_paths = [wind_file]
    else:
        dataset_paths = list_era5_datasets(data_dir)
        if not dataset_paths:
            typer.echo(
                f"[ошибка] В каталоге {data_dir} нет файлов *.nc.\n"
                "Скачайте данные командой: diplom download",
                err=True,
            )
            raise typer.Exit(code=1)

    missing = [p for p in dataset_paths if not p.exists()]
    if missing:
        for path in missing:
            typer.echo(f"[ошибка] Файл ERA5 не найден: {path}", err=True)
        raise typer.Exit(code=1)

    if list_times:
        for path in dataset_paths:
            available = list_available_times(path)
            typer.echo(f"Доступные временны́е метки в {path.name}:")
            for t in available:
                typer.echo(f"  {t}")
        return

    time_ns: int | None = None
    if time is not None:
        time_ns = int(np.datetime64(time).astype("datetime64[ns]").astype(np.int64))

    jobs: list[WindPlotRenderJob] = []
    for dataset_path in dataset_paths:
        plot_path = wind_plot_html_path(dataset_path, output)
        if plot_path.exists():
            typer.echo(f"Пропуск {dataset_path.name}: график уже есть → {plot_path}")
            continue
        jobs.append(
            WindPlotRenderJob(
                dataset_path=dataset_path,
                output_dir=output,
                time_ns=time_ns,
                stride_lon=stride_lon,
                stride_lat=stride_lat,
                stride_altitude_m=stride_altitude_m,
                w_scale=w_scale,
            )
        )

    if not jobs:
        typer.echo("Новых графиков не создано (все уже есть или нет датасетов).")
        return

    n_workers = workers if workers is not None else min(len(jobs), os.cpu_count() or 1)
    if n_workers > 1:
        typer.echo(f"Параллельная отрисовка: {len(jobs)} график(ов), workers={n_workers}")

    results = render_wind_plots(jobs, workers=n_workers)
    saved_paths: list[Path] = []
    errors: list[str] = []

    for result in results:
        for line in result.log_lines:
            typer.echo(line)
        if result.error:
            errors.append(result.error)
        elif result.saved and result.plot_path is not None:
            saved_paths.append(result.plot_path)

    if errors:
        for msg in errors:
            typer.echo(f"[ошибка] {msg}", err=True)
        raise typer.Exit(code=1)

    if not saved_paths:
        typer.echo("Новых графиков не создано (все уже есть или нет датасетов).")
        return

    if open_browser and len(saved_paths) == 1:
        webbrowser.open(saved_paths[0].resolve().as_uri())


# ──────────────────── Точка входа ────────────────────

def main() -> None:
    """Точка входа CLI."""
    app()


if __name__ == "__main__":
    main()
