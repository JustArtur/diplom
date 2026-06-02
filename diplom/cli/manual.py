from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Optional
import webbrowser

import numpy as np
import typer

from diplom.cli.training_options import (
    DATASET_OPTION,
    DATA_DIR_OPTION,
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
from diplom.envs.factory import build_env
from diplom.envs.constants import (
    TRAIN_TARGET_POSITION_HORIZONTAL_DELTA,
    TRAIN_TARGET_POSITION_VERTICAL_DELTA,
)
from diplom.envs.observations import get_obs_spec
from diplom.envs.rewards import get_reward_fn
from diplom.viz.plotly.episode_figure import (
    TRAJECTORY_LIVE_WIND_OVERLAY,
    build_placeholder_live_figure,
    collect_trajectory_traces,
)
from diplom.viz.plotly.trajectory import (
    EpisodeVizData,
    compute_trajectory_bounds,
    save_live_figure,
    save_live_trajectory_update,
)
from diplom.viz.plotly.episode_figure import build_wind_overlay_traces, wind_overlay_cache_key


def _create_run_dir(output_root: Path) -> Path:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_root / f"manual_{stamp}"
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = output_root / f"manual_{stamp}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _prompt_action_batch(action_limit: float) -> tuple[float, int] | None:
    prompt_text = (
        f"Введите action или action + steps одной строкой: "
        f"`<action>` или `<action> <steps>`, например `5` или `5 10` "
        f"(action в диапазоне [-{action_limit:.3f}, {action_limit:.3f}], "
        "steps >= 1, q/quit/exit — завершить эпизод)"
    )
    while True:
        raw = typer.prompt(prompt_text, default="0.0")
        text = raw.strip().lower()
        if text in {"q", "quit", "exit"}:
            return None
        parts = raw.replace(",", " ").split()
        if len(parts) not in {1, 2}:
            typer.echo("Нужно ввести одно или два значения: `<action>` или `<action> <steps>`")
            continue
        try:
            action = float(parts[0])
        except ValueError:
            typer.echo("Первое значение должно быть числом")
            continue
        if action < -action_limit or action > action_limit:
            typer.echo(
                f"action должен быть в диапазоне [-{action_limit:.3f}, {action_limit:.3f}]"
            )
            continue
        repeat_count = 1
        if len(parts) == 2:
            try:
                repeat_count = int(parts[1])
            except ValueError:
                typer.echo("Второе значение должно быть целым числом >= 1")
                continue
            if repeat_count < 1:
                typer.echo("Второе значение должно быть целым числом >= 1")
                continue
        return action, repeat_count


def _update_live_html(
    *,
    html_path: Path,
    env,
    current_steps: list[dict[str, object]],
    episode_idx: int,
    step_idx: int,
    live_wind_cones: bool,
    live_poll_ms: int,
) -> None:
    if not current_steps:
        return
    live_episode = EpisodeVizData(
        env_idx=0,
        steps=current_steps,
        target_position=np.asarray(
            current_steps[-1].get("target_position", [0.0, 0.0, 0.0]),
            dtype=np.float32,
        ),
        label=f"manual · эпизод {episode_idx + 1}",
    )
    bounds = compute_trajectory_bounds([live_episode], world_bounds=env.world_bounds)
    save_live_trajectory_update(
        html_path,
        generation=step_idx + 1,
        trajectory_traces=collect_trajectory_traces(
            env_idx=0,
            history=[],
            current_steps=current_steps,
            live_step_count=len(current_steps),
        ),
        title=(
            f"manual · эпизод {episode_idx + 1} · "
            f"шаг {step_idx + 1}"
        ),
        bounds=bounds,
        wind_traces=(
                build_wind_overlay_traces(
                    env.wind_interp,
                    env.sim.sim_time,
                    **TRAJECTORY_LIVE_WIND_OVERLAY,
                )
            if live_wind_cones
            else None
        ),
        wind_key=(
            wind_overlay_cache_key(env.sim.sim_time)
            if live_wind_cones
            else None
        ),
        poll_interval_ms=live_poll_ms,
    )


def _print_step_state(
    *,
    episode_idx: int,
    step_idx: int,
    action: float,
    reward: float,
    total_reward: float,
    obs: np.ndarray,
    info: dict[str, object],
    env_render: str,
    sim_time: object,
) -> None:
    typer.echo("")
    typer.echo(f"Эпизод {episode_idx + 1} | шаг {step_idx + 1}")
    typer.echo(f"  action                : {action:+.4f}")
    typer.echo(f"  reward                : {reward:+.4f}")
    typer.echo(f"  total_reward          : {total_reward:+.4f}")
    typer.echo(f"  sim_time              : {sim_time}")
    typer.echo(f"  obs_dim               : {obs.size}")
    typer.echo(f"  terminated            : {bool(info.get('terminated', False))}")
    typer.echo(f"  truncated             : {bool(info.get('truncated', False))}")
    typer.echo(f"  position_x            : {float(obs[0]) if obs.size > 0 else 0.0:+.4f}")
    typer.echo(f"  position_y            : {float(obs[1]) if obs.size > 1 else 0.0:+.4f}")
    typer.echo(f"  position_z            : {float(obs[2]) if obs.size > 2 else 0.0:+.4f}")
    typer.echo(f"  target_x              : {float(obs[3]) if obs.size > 3 else 0.0:+.4f}")
    typer.echo(f"  target_y              : {float(obs[4]) if obs.size > 4 else 0.0:+.4f}")
    typer.echo(f"  target_z              : {float(obs[5]) if obs.size > 5 else 0.0:+.4f}")
    typer.echo(f"  wind_u                : {float(obs[9]) if obs.size > 9 else 0.0:+.4f}")
    typer.echo(f"  wind_v                : {float(obs[10]) if obs.size > 10 else 0.0:+.4f}")
    typer.echo(f"  wind_w                : {float(obs[11]) if obs.size > 11 else 0.0:+.4f}")
    typer.echo(f"  energy_spent          : {float(obs[12]) if obs.size > 12 else 0.0:+.4f}")
    typer.echo(f"  air_weight            : {float(obs[13]) if obs.size > 13 else 0.0:+.4f}")
    typer.echo(f"  vertical_speed        : {float(obs[14]) if obs.size > 14 else 0.0:+.4f}")
    typer.echo(f"  vertical_acceleration : {float(obs[15]) if obs.size > 15 else 0.0:+.4f}")
    typer.echo(f"  air_density           : {float(obs[16]) if obs.size > 16 else 0.0:+.4f}")
    typer.echo(f"  temperature           : {float(obs[17]) if obs.size > 17 else 0.0:+.4f}")
    typer.echo(f"  pressure              : {float(obs[18]) if obs.size > 18 else 0.0:+.4f}")
    typer.echo(f"  wind_toward           : {float(info.get('wind_toward', 0.0)):+.4f}")
    typer.echo(f"  wind_align_delta      : {float(info.get('wind_align_delta', 0.0)):+.4f}")
    typer.echo(f"  horizontal_progress   : {float(info.get('horizontal_progress', 0.0)):+.4f}")
    typer.echo(f"  progress_reward       : {float(info.get('progress_reward', 0.0)):+.4f}")
    typer.echo(f"  distance_to_target    : {float(info.get('distance_to_target', 0.0)):.2f}")
    typer.echo(f"  horizontal_distance   : {float(info.get('horizontal_distance', 0.0)):.2f}")
    typer.echo(f"  reward_wind_align     : {float(info.get('reward_wind_align_term', 0.0)):+.4f}")
    typer.echo(f"  reward_wind_delta     : {float(info.get('reward_wind_align_delta_term', 0.0)):+.4f}")
    typer.echo(f"  reward_progress       : {float(info.get('reward_progress_term', 0.0)):+.4f}")
    typer.echo(f"  reward_goal           : {float(info.get('reward_goal_term', 0.0)):+.4f}")
    typer.echo(f"  reward_distance       : {float(info.get('reward_distance_term', 0.0)):+.4f}")
    typer.echo(f"  reward_energy         : {float(info.get('reward_energy_term', 0.0)):+.4f}")
    typer.echo(f"  reward_boundary       : {float(info.get('reward_boundary_term', 0.0)):+.4f}")
    typer.echo(f"  reward_best_distance  : {float(info.get('reward_best_distance_term', 0.0)):+.4f}")
    typer.echo(f"  reward_regression     : {float(info.get('reward_distance_regression_term', 0.0)):+.4f}")
    typer.echo(f"  reward_hold_close     : {float(info.get('reward_hold_close_term', 0.0)):+.4f}")
    typer.echo(f"  reward_wind_streak    : {float(info.get('reward_wind_streak_term', 0.0)):+.4f}")
    typer.echo(f"  reward_wind_adverse   : {float(info.get('reward_wind_adverse_streak_term', 0.0)):+.4f}")
    typer.echo(f"  reward_wind_scan      : {float(info.get('reward_wind_scan_term', 0.0)):+.4f}")
    typer.echo(f"  reward_adverse_close  : {float(info.get('reward_adverse_wind_close_term', 0.0)):+.4f}")
    typer.echo(f"  reward_high_altitude  : {float(info.get('reward_high_altitude_term', 0.0)):+.4f}")
    typer.echo(f"  reward_idle_action    : {float(info.get('reward_idle_action_term', 0.0)):+.4f}")
    typer.echo(f"  reward_z_stick        : {float(info.get('reward_z_stick_term', 0.0)):+.4f}")
    typer.echo(f"  {env_render}")


def manual_rollout(
    output: Path = typer.Option(
        Path("runs/manual"),
        "--output",
        "-o",
        help="Родительский каталог для интерактивных сессий; каждая сессия получает свой подкаталог",
    ),
    episodes: int = typer.Option(
        1,
        "--episodes",
        "-n",
        min=1,
        help="Сколько эпизодов пройти вручную за один запуск",
    ),
    seed: int = typer.Option(0, "--seed", help="Seed для воспроизводимости"),
    target_reach_radius: float = TARGET_RADIUS_OPTION,
    start_time: Optional[datetime] = START_TIME_OPTION,
    randomize_initial_position: bool = RANDOMIZE_INITIAL_POSITION_OPTION,
    randomize_target_position: bool = RANDOMIZE_TARGET_POSITION_OPTION,
    target_horizontal_delta: float = RANDOMIZE_TARGET_HORIZONTAL_DELTA_OPTION,
    target_vertical_delta: float = RANDOMIZE_TARGET_VERTICAL_DELTA_OPTION,
    dataset: str | None = DATASET_OPTION,
    data_dir: Path = DATA_DIR_OPTION,
    manifest_path: Path = MANIFEST_OPTION,
    open_browser: bool = typer.Option(
        True,
        "--open-browser/--no-open-browser",
        help="Открыть live HTML с траекторией в браузере",
    ),
    live_poll_ms: int = typer.Option(
        1000,
        "--live-poll-ms",
        min=250,
        help="Как часто браузер проверяет обновление траектории в HTML",
    ),
    live_wind_cones: bool = typer.Option(
        False,
        "--live-wind-cones/--no-live-wind-cones",
        help="Показывать конусы ветра в live HTML во время ручной симуляции",
    ),
    reward: str = REWARD_OPTION,
    obs: str = OBS_OPTION,
) -> None:
    """Интерактивный терминальный эпизод: человек выбирает action, среда пишет траекторию."""
    get_reward_fn(reward)
    get_obs_spec(obs)

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
    run_dir = _create_run_dir(output)
    trajectories_dir = run_dir / "trajectories"
    effective_randomize_target_position = bool(
        randomize_target_position
        or target_horizontal_delta != TRAIN_TARGET_POSITION_HORIZONTAL_DELTA
        or target_vertical_delta != TRAIN_TARGET_POSITION_VERTICAL_DELTA
    )
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
            trajectory_steps_dir=trajectories_dir,
            trajectory_record_observation=True,
        ),
    )

    env = build_env(app_config.environment, app_config.wind, env_idx=0)
    html_path = trajectories_dir / "trajectories.html"
    if open_browser:
        placeholder = build_placeholder_live_figure(0, env.world_bounds)
        if live_wind_cones:
            save_live_trajectory_update(
                html_path,
                generation=0,
                trajectory_traces=[],
                title="manual · ожидание данных…",
                bounds=compute_trajectory_bounds([], world_bounds=env.world_bounds),
                wind_traces=[],
                wind_key=0,
                poll_interval_ms=live_poll_ms,
            )
        else:
            save_live_figure(placeholder, html_path, generation=0, poll_interval_ms=live_poll_ms)
        webbrowser.open(html_path.resolve().as_uri())
    typer.echo(f"Run directory: {run_dir.resolve()}")
    typer.echo(f"Trajectories: {trajectories_dir.resolve()}")
    typer.echo(f"Live HTML: {html_path.resolve()}")
    typer.echo(f"Wind dataset: {app_config.wind.path.resolve()}")
    typer.echo(
        f"Reward={reward} Obs={obs} "
        f"randomize_initial_position={randomize_initial_position} "
        f"randomize_target_position={effective_randomize_target_position} "
        f"target_horizontal_delta={target_horizontal_delta} "
        f"target_vertical_delta={target_vertical_delta}"
    )

    try:
        for episode_idx in range(episodes):
            obs_vec, _ = env.reset(seed=seed + episode_idx)
            episode_reward = 0.0
            step_idx = 0
            current_steps: list[dict[str, object]] = []
            typer.echo("")
            typer.echo("=" * 80)
            typer.echo(f"Старт эпизода {episode_idx + 1}/{episodes}")
            typer.echo(f"Initial state: {env.render()}")
            typer.echo(f"Observation dim: {obs_vec.size}")

            while True:
                batch = _prompt_action_batch(float(env.action_limit))
                if batch is None:
                    typer.echo("Эпизод остановлен пользователем.")
                    return
                action, repeat_count = batch

                for repeat_idx in range(repeat_count):
                    next_obs, reward_value, terminated, truncated, info = env.step(
                        np.asarray([action], dtype=np.float32)
                    )
                    step_record = env.consume_step_record()
                    if step_record:
                        current_steps.append(step_record)
                        if repeat_count == 1:
                            _update_live_html(
                                html_path=html_path,
                                env=env,
                                current_steps=current_steps,
                                episode_idx=episode_idx,
                                step_idx=step_idx,
                                live_wind_cones=live_wind_cones,
                                live_poll_ms=live_poll_ms,
                            )
                    episode_reward += float(reward_value)
                    _print_step_state(
                        episode_idx=episode_idx,
                        step_idx=step_idx,
                        action=action,
                        reward=float(reward_value),
                        total_reward=episode_reward,
                        obs=obs_vec,
                        info=info,
                        env_render=env.render(),
                        sim_time=env.sim.sim_time,
                    )
                    obs_vec = next_obs
                    step_idx += 1
                    if terminated or truncated:
                        typer.echo(
                            f"Эпизод завершён: terminated={terminated} truncated={truncated} "
                            f"steps={step_idx} total_reward={episode_reward:+.4f}"
                        )
                        break
                if repeat_count > 1 and current_steps:
                    _update_live_html(
                        html_path=html_path,
                        env=env,
                        current_steps=current_steps,
                        episode_idx=episode_idx,
                        step_idx=step_idx - 1,
                        live_wind_cones=live_wind_cones,
                        live_poll_ms=live_poll_ms,
                    )
                if terminated or truncated:
                    break

            if episode_idx + 1 < episodes and not typer.confirm("Начать следующий эпизод?"):
                break
    except KeyboardInterrupt:
        typer.echo("\nПрервано пользователем.")
    finally:
        env.close()
