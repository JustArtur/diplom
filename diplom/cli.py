"""CLI-интерфейс для симулятора стратостата и RL-обучения."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from datetime import datetime
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
    target_reach_radius: float = typer.Option(
        DEFAULT_ENVIRONMENT_CONFIG.target_reach_radius,
        "--target-radius",
        help="Радиус вокруг цели, при попадании в который эпизод считается успешным",
    ),
    start_time: datetime = typer.Option(
        datetime.fromisoformat(str(DEFAULT_VISUALIZATION_CONFIG.sim_start_time)),
        help="Базовое время симуляции; в train-эпизодах с рандомизацией будет переопределено.",
    ),
) -> None:
    """Запустить обучение PPO-модели."""
    from diplom.train.ppo_runner import train_ppo as run_train_ppo

    # Верхний слой задаёт только пользовательские параметры обучения.
    app_config = _build_app_config(
        total_timesteps=total_timesteps,
        seed=seed,
        logdir=logdir,
        n_envs=n_envs,
        target_reach_radius=target_reach_radius,
        start_time=start_time,
    )
    run_train_ppo(app_config)

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
) -> None:
    """Запустить rollout обученной модели и собрать траектории эпизодов."""
    from diplom.sim.rollout import rollout_episodes

    app_config = AppConfig(
        environment=EnvironmentConfig(
            balloon=BalloonConfig(sim_time=np.datetime64(start_time)),
            target_reach_radius=target_reach_radius,
        ),
    )
    results = rollout_episodes(
        app_config,
        n_episodes=episodes,
        policy_path=str(model_path),
        render=render,
        seed=seed,
    )

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
        bounds = compute_trajectory_bounds(viz_episodes)
        fig = build_figure(
            episodes=viz_episodes,
            title=f"Rollout · {model_path.name} · {episodes} эпизодов",
            bounds=bounds,
        )
        save_figure(fig, plot_output)
        typer.echo(f"plot saved={plot_output}")


# ──────────────────── Точка входа ────────────────────

def main() -> None:
    """Точка входа CLI."""
    app()


if __name__ == "__main__":
    main()
