"""CLI-интерфейс для симулятора стратостата и RL-обучения."""

from __future__ import annotations

import json
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


def _build_app_config(
    *,
    total_timesteps: int = DEFAULT_TRAINING_CONFIG.total_timesteps,
    seed: int = DEFAULT_TRAINING_CONFIG.seed,
    logdir: Path = DEFAULT_TRAINING_CONFIG.logdir,
    n_envs: int = DEFAULT_TRAINING_CONFIG.n_envs,
    device: str = DEFAULT_TRAINING_CONFIG.device,
    verbose: int = DEFAULT_TRAINING_CONFIG.verbose,
    target_reach_radius: float = DEFAULT_ENVIRONMENT_CONFIG.target_reach_radius,
    start_time: datetime = datetime.fromisoformat(str(DEFAULT_VISUALIZATION_CONFIG.sim_start_time)),
) -> AppConfig:
    return AppConfig(
        environment=EnvironmentConfig(
            balloon=BalloonConfig(sim_time=np.datetime64(start_time)),
            target_reach_radius=target_reach_radius,
        ),
        training=TrainingConfig(
            total_timesteps=total_timesteps,
            seed=seed,
            logdir=logdir,
            n_envs=n_envs,
            device=device,
            verbose=verbose,
        ),
        visualization=VisualizationConfig(
            window_size=DEFAULT_VISUALIZATION_CONFIG.window_size,
            bg_bottom=DEFAULT_VISUALIZATION_CONFIG.bg_bottom,
            bg_top=DEFAULT_VISUALIZATION_CONFIG.bg_top,
            sim_start_time=np.datetime64(start_time),
        ),
    )


# ──────────────────── download ────────────────────

@app.command()
def download(
    outfile: Path = typer.Option(DEFAULT_DOWNLOAD_CONFIG.outfile, "--outfile", "-o"),
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
) -> None:
    """Скачать подмножество ERA5 в NetCDF."""
    from diplom.data.era5_download import download_era5_pressure

    # CLI лишь собирает конфиг и передаёт его вниз.
    download_era5_pressure(
        DownloadConfig(
            outfile=outfile,
            north=north,
            west=west,
            south=south,
            east=east,
            start=start,
            end=end,
            pressure_levels=tuple(level),
            variables=tuple(variable),
        )
    )

# ──────────────────── viz_real ────────────────────

@app.command()
def viz_real(
    start_time: datetime = typer.Option(
        datetime.fromisoformat(str(DEFAULT_VISUALIZATION_CONFIG.sim_start_time)),
        help="Время слоя симуляции",
    ),
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
    start_time: datetime = typer.Option(
        datetime.fromisoformat(str(DEFAULT_VISUALIZATION_CONFIG.sim_start_time)),
        help="Базовое время симуляции; в train-эпизодах с рандомизацией будет переопределено.",
    ),
    in_process: bool = typer.Option(
        False,
        "--in-process",
        help="Одна среда в DummyVecEnv (для profile-ppo-mem/cpu; не использовать в боевом обучении)",
    ),
    verbose: int = typer.Option(
        DEFAULT_TRAINING_CONFIG.verbose,
        "--verbose",
        "-v",
        help="Уровень логирования PPO в консоль (SB3): 0 — тихо, 1 — таблица метрик; env-метрики только в TensorBoard",
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
        target_reach_radius=target_reach_radius,
        start_time=start_time,
    )
    try:
        run_train_ppo(app_config, force_dummy_vec_env=in_process)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


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
    start_time: datetime = typer.Option(
        datetime.fromisoformat(str(DEFAULT_VISUALIZATION_CONFIG.sim_start_time)),
        help="Базовое время симуляции",
    ),
) -> None:
    """Профиль памяти при обучении PPO (memray): одна среда, один процесс.

    CPU (время): diplom profile-ppo-cpu. Запускайте отдельно — совмещение сильно замедляет прогон.

    Установка: poetry install --with dev

    \b
      diplom profile-ppo-mem -t 50000
      diplom profile-ppo-mem -t 50000 --native
      open runs/profile_ppo/PPO_0/memray.html
    """
    from diplom.train.profiling import PROFILE_N_ENVS, MemrayNotFoundError, run_memray_train

    app_config = _build_app_config(
        total_timesteps=total_timesteps,
        seed=seed,
        logdir=logdir,
        n_envs=PROFILE_N_ENVS,
        device=device,
        verbose=verbose,
        target_reach_radius=target_reach_radius,
        start_time=start_time,
    )
    try:
        bin_path, html_path = run_memray_train(
            app_config,
            output=output,
            flamegraph=flamegraph,
            skip_flamegraph=no_flamegraph,
            native_traces=native,
            print_table=not no_table,
        )
    except MemrayNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Run: {bin_path.parent}")
    if html_path is not None:
        typer.echo(f"Flame graph: {html_path}")


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
    start_time: datetime = typer.Option(
        datetime.fromisoformat(str(DEFAULT_VISUALIZATION_CONFIG.sim_start_time)),
    ),
) -> None:
    """Профиль CPU (cProfile): одна среда, один процесс.

    Память: diplom profile-ppo-mem (memray). Запускайте отдельно от profile-ppo-cpu.

    \b
      diplom profile-ppo-cpu -t 50000
      snakeviz runs/profile_ppo/PPO_0/profile.prof
    """
    from diplom.train.profiling import PROFILE_N_ENVS, run_cprofile_train

    app_config = _build_app_config(
        total_timesteps=total_timesteps,
        seed=seed,
        logdir=logdir,
        n_envs=PROFILE_N_ENVS,
        device=device,
        verbose=verbose,
        target_reach_radius=target_reach_radius,
        start_time=start_time,
    )
    try:
        prof_path = run_cprofile_train(app_config, output=output, top_lines=top_lines)
        typer.echo(f"Run: {prof_path.parent}")
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


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
    start_time: datetime = typer.Option(
        datetime.fromisoformat(str(DEFAULT_VISUALIZATION_CONFIG.sim_start_time)),
        help="Базовое время симуляции",
    ),
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
            balloon=BalloonConfig(sim_time=np.datetime64(start_time)),
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
    wind_file: Path = typer.Option(
        Path("data/era5_sample.nc"),
        "--wind-file", "-f",
        help="Путь к ERA5 NetCDF-файлу",
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
        Path("runs/wind/wind_field.html"),
        "--output", "-o",
        help="Куда сохранить HTML-файл с графиком",
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
) -> None:
    """Построить интерактивный 3D-граф поля ветра ERA5.

    Конусы показывают направление и скорость ветра на всех высотах,
    широтах и долготах для выбранного временного среза ERA5.

    Примеры:

    \b
      # Список доступных временных меток
      diplom wind-viz --list-times

    \b
      # График на конкретное время с прореживанием
      diplom wind-viz --time 2024-07-01T12:00:00 --stride-lat 2 --stride-lon 2 --stride-altitude-m 1000
    """
    from diplom.viz.wind_plot import (
        build_wind_figure,
        list_available_times,
        load_wind_slice,
        save_figure,
    )

    if not wind_file.exists():
        typer.echo(
            f"[ошибка] Файл ERA5 не найден: {wind_file}\n"
            "Скачайте данные командой: diplom download",
            err=True,
        )
        raise typer.Exit(code=1)

    if list_times:
        available = list_available_times(wind_file)
        typer.echo(f"Доступные временны́е метки в {wind_file}:")
        for t in available:
            typer.echo(f"  {t}")
        return

    # Если время не задано — берём первый шаг датасета
    target_time: np.datetime64
    if time is None:
        available = list_available_times(wind_file)
        if not available:
            typer.echo("[ошибка] Датасет не содержит временны́х шагов.", err=True)
            raise typer.Exit(code=1)
        target_time = available[0]
        typer.echo(f"--time не задано, используется первый шаг: {target_time}")
    else:
        target_time = np.datetime64(time)

    typer.echo(f"Загружаю срез ERA5: {wind_file} @ {target_time} …")
    wind_slice = load_wind_slice(wind_file, target_time)
    typer.echo(
        f"Срез загружен · время={wind_slice.time} "
        f"· уровней={len(wind_slice.pressure)} "
        f"· lat={len(wind_slice.lat)} · lon={len(wind_slice.lon)}"
    )

    from diplom.world import log_world_bounds, world_bounds_from_axes

    wb = world_bounds_from_axes(
        np.asarray(wind_slice.lat, dtype=np.float64),
        np.asarray(wind_slice.lon, dtype=np.float64),
        origin_lat=wind_slice.origin_lat,
        origin_lon=wind_slice.origin_lon,
        pressure_axis_hpa=np.asarray(wind_slice.pressure, dtype=np.float64),
    )
    log_world_bounds(
        wb,
        origin_lat=wind_slice.origin_lat,
        origin_lon=wind_slice.origin_lon,
        wind_path=wind_file,
        prefix="[wind-viz]",
    )

    fig = build_wind_figure(
        wind_slice,
        stride_lon=stride_lon,
        stride_lat=stride_lat,
        stride_altitude_m=stride_altitude_m,
        w_scale=w_scale,
    )

    save_figure(fig, output)
    typer.echo(f"График сохранён: {output}")

    if open_browser:
        webbrowser.open(output.resolve().as_uri())


# ──────────────────── Точка входа ────────────────────

def main() -> None:
    """Точка входа CLI."""
    app()


if __name__ == "__main__":
    main()
