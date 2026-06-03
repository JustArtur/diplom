from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Optional

import json
import typer

from diplom.cli.defaults import DEFAULT_ROLLOUT_MODEL_PATH, DEFAULT_TRAINING_CONFIG
from diplom.cli.training_options import (
    OBS_OPTION,
    REWARD_OPTION,
    START_TIME_OPTION,
    TARGET_RADIUS_OPTION,
    balloon_config,
)
from diplom.config import AppConfig, EnvironmentConfig
from diplom.envs.observations import get_obs_spec
from diplom.envs.rewards import get_reward_fn
from diplom.wind.factory import build_wind_interpolator


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
    target_reach_radius: float = TARGET_RADIUS_OPTION,
    plot_output: Path | None = typer.Option(
        None,
        "--plot",
        "-p",
        help="Путь для сохранения интерактивного 3D-графика траекторий (HTML). "
             "Если не указан, график не строится.",
    ),
    plot_wind_cones: bool = typer.Option(
        False,
        "--plot-wind-cones/--no-plot-wind-cones",
        help="Показать конусы ветра на графике траекторий (нужен --plot)",
    ),
    start_time: Optional[datetime] = START_TIME_OPTION,
    device: str = typer.Option(
        DEFAULT_TRAINING_CONFIG.device,
        "--device", "-d",
        help="Устройство для загрузки PPO: cpu, cuda или mps",
        case_sensitive=False,
    ),
    reward: str = REWARD_OPTION,
    obs: str = OBS_OPTION,
) -> None:
    # Запустить rollout обученной модели и собрать траектории эпизодов.
    get_reward_fn(reward)
    get_obs_spec(obs)
    from diplom.sim.rollout import rollout_episodes
    from diplom.viz.plotly.episode_figure import (
        rollout_results_to_episodes,
        save_rollout_figure,
    )

    app_config = AppConfig(
        environment=EnvironmentConfig(
            balloon=balloon_config(start_time),
            target_reach_radius=target_reach_radius,
            reward_name=reward,
            obs_name=obs,
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
        "reward": reward,
        "obs": obs,
        "results": serialized_results,
    }

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    typer.echo(f"reward={reward} obs={obs}")
    for idx, result in enumerate(results, start=1):
        typer.echo(
            f"episode={idx} success={result.success} steps={result.steps} total_reward={result.total_reward:.3f}"
        )

    if output is not None:
        typer.echo(f"saved={output}")

    if plot_output is not None:
        viz_episodes = rollout_results_to_episodes(results)
        wind_interpolator = build_wind_interpolator(app_config.wind)
        try:
            save_rollout_figure(
                viz_episodes,
                title=f"Rollout · {model_path.name} · {episodes} эпизодов",
                wind_interpolator=wind_interpolator,
                output_path=plot_output,
                show_wind_cones=plot_wind_cones,
            )
        finally:
            wind_interpolator.close()
        typer.echo(f"plot saved={plot_output}")
