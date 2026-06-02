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
    RANDOMIZE_INITIAL_POSITION_OPTION,
    RANDOMIZE_TARGET_POSITION_OPTION,
    RANDOMIZE_TARGET_HORIZONTAL_DELTA_OPTION,
    RANDOMIZE_TARGET_VERTICAL_DELTA_OPTION,
    REWARD_OPTION,
    START_TIME_OPTION,
    TARGET_RADIUS_OPTION,
    build_default_app_config,
)
from diplom.data.era5_paths import resolve_dataset_reference
from diplom.envs.constants import (
    TRAIN_TARGET_POSITION_HORIZONTAL_DELTA,
    TRAIN_TARGET_POSITION_VERTICAL_DELTA,
)
from diplom.envs.observations import get_obs_spec
from diplom.envs.rewards import get_reward_fn
from diplom.sim.greedy import DEFAULT_GREEDY_RUN_CONFIG, GreedyRunConfig, greedy_episodes
from diplom.trajectory.steps_io import cleanup_steps_dir


def _create_run_dir(output_root: Path) -> Path:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_root / f"greedy_{stamp}"
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = output_root / f"greedy_{stamp}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def greedy(
    episodes: int = typer.Option(1, "--episodes", "-n", min=1, help="Количество эпизодов"),
    seed: int = typer.Option(DEFAULT_TRAINING_CONFIG.seed, "--seed", help="Seed для воспроизводимости"),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Куда сохранить результаты greedy в JSON",
    ),
    render: bool = typer.Option(
        False,
        "--render/--no-render",
        help="Печатать состояние среды на каждом шаге",
    ),
    target_reach_radius: float = TARGET_RADIUS_OPTION,
    randomize_initial_position: bool = RANDOMIZE_INITIAL_POSITION_OPTION,
    randomize_target_position: bool = RANDOMIZE_TARGET_POSITION_OPTION,
    target_horizontal_delta: float = RANDOMIZE_TARGET_HORIZONTAL_DELTA_OPTION,
    target_vertical_delta: float = RANDOMIZE_TARGET_VERTICAL_DELTA_OPTION,
    dataset: str | None = DATASET_OPTION,
    data_dir: Path = DATA_DIR_OPTION,
    manifest_path: Path = MANIFEST_OPTION,
    trajectories: bool = typer.Option(
        True,
        "--trajectories/--no-trajectories",
        help="Live HTML-траектории и JSONL шагов (как в train-ppo)",
    ),
    open_trajectories: bool = typer.Option(
        False,
        "--open-trajectories/--no-open-trajectories",
        help="Открыть live-viewer trajectories.html в браузере при старте (нужны --trajectories)",
    ),
    trajectory_wind_cones: bool = typer.Option(
        False,
        "--trajectory-wind-cones/--no-trajectory-wind-cones",
        help="Конусы ветра на live HTML-графике (нужны --trajectories)",
    ),
    trajectory_render_interval: int = typer.Option(
        DEFAULT_GREEDY_RUN_CONFIG.trajectory_render_interval,
        "--trajectory-render-interval",
        min=1,
        help="Как часто обновлять live HTML (каждые N шагов симуляции)",
    ),
    plot: bool = typer.Option(
        False,
        "--plot/--no-plot",
        help="Дополнительно сохранить статичный HTML с траекториями после прогона",
    ),
    plot_output: Path | None = typer.Option(
        None,
        "--plot-output",
        "-p",
        help="Путь для статичного HTML (нужен --plot)",
    ),
    plot_wind_cones: bool = typer.Option(
        False,
        "--plot-wind-cones/--no-plot-wind-cones",
        help="Конусы ветра на статичном HTML (нужен --plot)",
    ),
    start_time: Optional[datetime] = START_TIME_OPTION,
    reward: str = REWARD_OPTION,
    obs: str = OBS_OPTION,
    lookahead_steps: int = typer.Option(
        DEFAULT_GREEDY_RUN_CONFIG.lookahead_steps,
        "--lookahead-steps",
        min=1,
        help="Сколько шагов вперёд симулировать для оценки каждого кандидата",
    ),
    candidate_actions: int = typer.Option(
        DEFAULT_GREEDY_RUN_CONFIG.candidate_count,
        "--candidate-actions",
        min=3,
        help="Сколько кандидатов действия проверять на каждом шаге",
    ),
    vertical_weight: float = typer.Option(
        DEFAULT_GREEDY_RUN_CONFIG.vertical_weight,
        "--vertical-weight",
        min=0.0,
        help="Вес вертикальной дистанции в greedy-оценке",
    ),
) -> None:
    """Запустить жадный контроллер и собрать траектории эпизодов."""
    get_reward_fn(reward)
    get_obs_spec(obs)

    if open_trajectories and not trajectories:
        typer.echo(
            "[ошибка] --open-trajectories требует включённых --trajectories",
            err=True,
        )
        raise typer.Exit(code=1)
    if plot and plot_output is None and output is not None:
        plot_output = output.with_suffix(".html")

    app_config = build_default_app_config(start_time=start_time)
    if dataset is not None:
        app_config = replace(
            app_config,
            wind=replace(
                app_config.wind,
                path=resolve_dataset_reference(
                    dataset,
                    data_dir=data_dir,
                    manifest_path=manifest_path,
                ),
            ),
        )
    effective_randomize_target_position = bool(
        randomize_target_position
        or target_horizontal_delta != TRAIN_TARGET_POSITION_HORIZONTAL_DELTA
        or target_vertical_delta != TRAIN_TARGET_POSITION_VERTICAL_DELTA
    )

    if output is not None:
        run_dir = Path(output).resolve().parent
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = _create_run_dir(Path("runs/greedy"))

    traj_dir = run_dir / "trajectories" if trajectories else None
    if traj_dir is not None:
        cleanup_steps_dir(traj_dir)

    app_config = replace(
        app_config,
        environment=replace(
            app_config.environment,
            target_reach_radius=target_reach_radius,
            randomize_initial_position=randomize_initial_position,
            randomize_target_position=effective_randomize_target_position,
            train_target_position_horizontal_delta=target_horizontal_delta,
            train_target_position_vertical_delta=target_vertical_delta,
            reward_name=reward,
            obs_name=obs,
            trajectory_steps_dir=traj_dir,
            trajectory_show_wind_cones=trajectory_wind_cones,
            trajectory_combined_html=True,
        ),
    )

    greedy_cfg = GreedyRunConfig(
        lookahead_steps=lookahead_steps,
        candidate_count=candidate_actions,
        vertical_weight=vertical_weight,
        trajectory_render_interval=trajectory_render_interval,
    )

    typer.echo(f"Run directory: {run_dir.resolve()}")
    if traj_dir is not None:
        typer.echo(f"Live trajectories: {traj_dir.resolve()}")
    else:
        typer.echo("Live trajectories: off (--no-trajectories)")
    typer.echo(f"Wind dataset: {app_config.wind.path.resolve()}")
    typer.echo(f"reward={reward} obs={obs}")
    typer.echo(
        f"randomize_initial_position={randomize_initial_position} "
        f"randomize_target_position={effective_randomize_target_position} "
        f"target_horizontal_delta={target_horizontal_delta} "
        f"target_vertical_delta={target_vertical_delta}"
    )
    typer.echo(
        f"strategy=greedy lookahead_steps={lookahead_steps} "
        f"candidate_actions={candidate_actions} vertical_weight={vertical_weight}"
    )

    results = greedy_episodes(
        app_config,
        n_episodes=episodes,
        greedy=greedy_cfg,
        render=render,
        seed=seed,
        open_trajectory_viz=open_trajectories,
    )

    serialized_results = [asdict(result) for result in results]
    summary = {
        "episodes": episodes,
        "seed": seed,
        "reward": reward,
        "obs": obs,
        "run_dir": str(run_dir.resolve()),
        "trajectories_dir": str(traj_dir.resolve()) if traj_dir is not None else None,
        "strategy": {
            "lookahead_steps": lookahead_steps,
            "candidate_actions": candidate_actions,
            "vertical_weight": vertical_weight,
            "trajectory_render_interval": trajectory_render_interval,
        },
        "results": serialized_results,
    }

    json_path = output if output is not None else run_dir / "greedy_results.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    for idx, result in enumerate(results, start=1):
        typer.echo(
            f"episode={idx} success={result.success} steps={result.steps} "
            f"total_reward={result.total_reward:.3f}"
        )
    typer.echo(f"saved={json_path.resolve()}")

    if plot:
        from diplom.viz.plotly.episode_figure import rollout_results_to_episodes, save_rollout_figure
        from diplom.wind.factory import build_wind_interpolator

        if plot_output is None:
            plot_output = run_dir / "trajectories_static.html"
        viz_episodes = rollout_results_to_episodes(results)
        wind_interpolator = build_wind_interpolator(app_config.wind)
        try:
            save_rollout_figure(
                viz_episodes,
                title=f"Greedy · {episodes} эпизодов",
                wind_interpolator=wind_interpolator,
                output_path=plot_output,
                show_wind_cones=plot_wind_cones,
            )
        finally:
            wind_interpolator.close()
        typer.echo(f"static plot saved={plot_output.resolve()}")
