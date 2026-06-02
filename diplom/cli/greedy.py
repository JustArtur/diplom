from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Optional

import json
import typer

from diplom.cli.defaults import DEFAULT_TRAINING_CONFIG
from diplom.cli.training_options import (
    DATA_DIR_OPTION,
    DATASET_OPTION,
    MANIFEST_OPTION,
    OBS_OPTION,
    REWARD_OPTION,
    START_TIME_OPTION,
    TARGET_RADIUS_OPTION,
    build_default_app_config,
)
from diplom.data.era5_paths import era5_dataset_title
from diplom.envs.observations import get_obs_spec
from diplom.envs.rewards import get_reward_fn
from diplom.sim.greedy import GreedyRunConfig, greedy_episodes

_DEFAULT_GREEDY = GreedyRunConfig()


def _default_trajectory_dir(wind_path: Path) -> Path:
    return Path("greedy") / era5_dataset_title(wind_path) / "trajectories"


def greedy(
    episodes: int = typer.Option(1, "--episodes", "-n", help="Количество эпизодов"),
    seed: int = typer.Option(DEFAULT_TRAINING_CONFIG.seed, "--seed", help="Seed для воспроизводимости"),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Куда сохранить результаты greedy baseline в JSON",
    ),
    render: bool = typer.Option(
        False,
        "--render/--no-render",
        help="Печатать состояние среды на каждом шаге",
    ),
    target_reach_radius: float = TARGET_RADIUS_OPTION,
    trajectories: bool = typer.Option(
        True,
        "--trajectories/--no-trajectories",
        help="HTML-траектории и JSONL шагов в каталоге trajectories (отключите для скорости)",
    ),
    trajectory_dir: Path | None = typer.Option(
        None,
        "--trajectory-dir",
        help="Каталог для trajectories.html и JSONL шагов (по умолчанию greedy/{dataset}/trajectories)",
    ),
    plot_output: Path | None = typer.Option(
        None,
        "--plot",
        "-p",
        help="Путь для HTML-графика траекторий (по умолчанию {trajectory-dir}/trajectories.html)",
    ),
    plot_wind_cones: bool = typer.Option(
        False,
        "--plot-wind-cones/--no-plot-wind-cones",
        help="Показать конусы ветра на HTML-графике траекторий",
    ),
    open_trajectories: bool = typer.Option(
        False,
        "--open-trajectories/--no-open-trajectories",
        help="Открыть live-viewer trajectories.html в браузере при старте (нужны --trajectories)",
    ),
    trajectory_render_interval: int = typer.Option(
        _DEFAULT_GREEDY.trajectory_render_interval,
        "--trajectory-render-interval",
        help="Обновлять live HTML каждые N шагов симуляции",
    ),
    start_time: Optional[datetime] = START_TIME_OPTION,
    dataset: Optional[str] = DATASET_OPTION,
    data_dir: Path = DATA_DIR_OPTION,
    manifest_path: Path = MANIFEST_OPTION,
    reward: str = REWARD_OPTION,
    obs: str = OBS_OPTION,
    lookahead_steps: int = typer.Option(
        _DEFAULT_GREEDY.lookahead_steps,
        "--lookahead-steps",
        help="Число шагов симуляции вперёд при оценке каждого кандидата действия",
    ),
    candidate_count: int = typer.Option(
        _DEFAULT_GREEDY.candidate_count,
        "--candidate-count",
        help="Число кандидатов действия на равномерной сетке [-action_limit, action_limit]",
    ),
    vertical_weight: float = typer.Option(
        _DEFAULT_GREEDY.vertical_weight,
        "--vertical-weight",
        help="Вес вертикальной дистанции до цели в score жадного выбора",
    ),
) -> None:
    """Запустить жадный baseline и собрать траектории эпизодов."""
    if open_trajectories and not trajectories:
        typer.echo(
            "[ошибка] --open-trajectories требует включённых --trajectories",
            err=True,
        )
        raise typer.Exit(code=1)

    get_reward_fn(reward)
    get_obs_spec(obs)

    app_config = build_default_app_config(
        start_time=start_time,
        dataset=dataset,
        data_dir=data_dir,
        manifest_path=manifest_path,
    )
    resolved_trajectory_dir = (
        trajectory_dir
        if trajectory_dir is not None
        else _default_trajectory_dir(app_config.wind.path)
    )
    app_config = replace(
        app_config,
        environment=replace(
            app_config.environment,
            target_reach_radius=target_reach_radius,
            reward_name=reward,
            obs_name=obs,
            trajectory_steps_dir=resolved_trajectory_dir if trajectories else None,
            trajectory_show_wind_cones=plot_wind_cones,
        ),
        training=replace(DEFAULT_TRAINING_CONFIG),
    )
    greedy_cfg = GreedyRunConfig(
        lookahead_steps=lookahead_steps,
        candidate_count=candidate_count,
        vertical_weight=vertical_weight,
        trajectory_render_interval=trajectory_render_interval,
    )

    try:
        results = greedy_episodes(
            app_config,
            n_episodes=episodes,
            greedy=greedy_cfg,
            render=render,
            seed=seed,
            open_trajectory_viz=open_trajectories,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    serialized_results = [asdict(result) for result in results]
    summary = {
        "episodes": episodes,
        "seed": seed,
        "policy": "greedy",
        "dataset": dataset,
        "wind_path": str(app_config.wind.path),
        "trajectory_dir": str(resolved_trajectory_dir) if trajectories else None,
        "reward": reward,
        "obs": obs,
        "greedy": asdict(greedy_cfg),
        "results": serialized_results,
    }

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    typer.echo(
        f"dataset={app_config.wind.path.name} reward={reward} obs={obs} "
        f"lookahead_steps={lookahead_steps} candidate_count={candidate_count}"
    )
    for idx, result in enumerate(results, start=1):
        typer.echo(
            f"episode={idx} success={result.success} steps={result.steps} total_reward={result.total_reward:.3f}"
        )

    if output is not None:
        typer.echo(f"saved={output}")

    if trajectories:
        typer.echo(
            f"trajectories dir={resolved_trajectory_dir.resolve()} "
            f"html={resolved_trajectory_dir / 'trajectories.html'}"
        )
    elif plot_output is not None:
        from diplom.viz.plotly.episode_figure import (
            rollout_results_to_episodes,
            save_rollout_figure,
        )
        from diplom.wind.factory import build_wind_interpolator

        viz_episodes = rollout_results_to_episodes(results)
        wind_interpolator = build_wind_interpolator(app_config.wind)
        try:
            save_rollout_figure(
                viz_episodes,
                title=(
                    f"Greedy baseline · {episodes} эпизодов · "
                    f"lookahead={lookahead_steps}"
                ),
                wind_interpolator=wind_interpolator,
                output_path=plot_output,
                show_wind_cones=plot_wind_cones,
            )
        finally:
            wind_interpolator.close()
        typer.echo(f"plot saved={plot_output}")
