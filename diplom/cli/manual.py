from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional
import time
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
from diplom.trajectory.replay import replay_episode_actions, rewrite_env_current_trajectory
from diplom.trajectory.smoothing import SmoothMethod, count_action_transitions, smooth_actions
from diplom.viz.plotly.episode_figure import (
    TRAJECTORY_LIVE_WIND_OVERLAY,
    collect_trajectory_traces,
)
from diplom.viz.plotly.trajectory import (
    EpisodeVizData,
    compute_trajectory_bounds,
    save_live_trajectory_update,
)
from diplom.viz.plotly.episode_figure import build_wind_overlay_traces, wind_overlay_cache_key


@dataclass(frozen=True, slots=True)
class _ManualCommand:
    kind: Literal["step", "smooth", "quit"]
    action: float = 0.0
    repeat_count: int = 1
    smooth_method: SmoothMethod | None = None
    smooth_blend_fraction: float | None = None
    smooth_range_start: int | None = None
    smooth_range_end: int | None = None
    smooth_alpha: float | None = None
    smooth_window: int | None = None


def _is_step_number(token: str) -> bool:
    try:
        value = int(token)
    except ValueError:
        return False
    return value >= 1


def _parse_smooth_command(parts: list[str]) -> _ManualCommand | str | None:
    if not parts:
        return _ManualCommand(kind="smooth")

    method: SmoothMethod | None = None
    blend_fraction: float | None = None
    range_start: int | None = None
    range_end: int | None = None
    alpha: float | None = None
    window: int | None = None

    idx = 0
    if idx < len(parts):
        token = parts[idx].lower()
        if token in {"transition", "ema", "moving_average", "ma"}:
            method = "moving_average" if token == "ma" else token  # type: ignore[assignment]
            idx += 1

    if idx + 1 < len(parts) and _is_step_number(parts[idx]) and _is_step_number(parts[idx + 1]):
        range_start = int(parts[idx])
        range_end = int(parts[idx + 1])
        idx += 2
    elif idx < len(parts):
        try:
            numeric = float(parts[idx].replace(",", "."))
        except ValueError:
            return f"Не удалось разобрать параметр сглаживания: {parts[idx]!r}"
        if method == "ema":
            alpha = numeric
        elif method == "moving_average":
            window = int(numeric)
        else:
            blend_fraction = numeric
        idx += 1
        if idx + 1 < len(parts) and _is_step_number(parts[idx]) and _is_step_number(parts[idx + 1]):
            range_start = int(parts[idx])
            range_end = int(parts[idx + 1])

    if range_start is not None and range_end is not None and range_start > range_end:
        range_start, range_end = range_end, range_start

    return _ManualCommand(
        kind="smooth",
        smooth_method=method,
        smooth_blend_fraction=blend_fraction,
        smooth_range_start=range_start,
        smooth_range_end=range_end,
        smooth_alpha=alpha,
        smooth_window=window,
    )


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


def _prompt_action_batch(
    action_limit: float,
    *,
    default_smooth_method: SmoothMethod,
    default_smooth_blend_fraction: float,
    default_smooth_alpha: float,
    default_smooth_window: int,
) -> _ManualCommand | None:
    prompt_text = (
        f"Введите action или action + steps одной строкой: "
        f"<action> или <action> <steps>, например 5 или 5 10 "
        f"(action в диапазоне [-{action_limit:.3f}, {action_limit:.3f}], "
        "steps >= 1; s, сглаживание с последнего s; "
        "s 500 1000, сглаживание шагов 500–1000; "
        "s 0.6 500 1000, с коэффициентом; q/quit/exit, завершить эпизод)"
    )
    while True:
        raw = typer.prompt(prompt_text, default="0.0")
        text = raw.strip().lower()
        if text in {"q", "quit", "exit"}:
            return _ManualCommand(kind="quit")
        if text in {"smooth", "s"} or text.startswith("smooth ") or text.startswith("s "):
            parts = raw.strip().split()[1:]
            parsed = _parse_smooth_command(parts)
            if isinstance(parsed, str):
                typer.echo(parsed)
                continue
            command = parsed or _ManualCommand(kind="smooth")
            return replace(
                command,
                smooth_method=command.smooth_method or default_smooth_method,
                smooth_blend_fraction=(
                    command.smooth_blend_fraction
                    if command.smooth_blend_fraction is not None
                    else default_smooth_blend_fraction
                ),
                smooth_alpha=command.smooth_alpha if command.smooth_alpha is not None else default_smooth_alpha,
                smooth_window=command.smooth_window or default_smooth_window,
            )
        parts = raw.replace(",", " ").split()
        if len(parts) not in {1, 2}:
            typer.echo("Нужно ввести одно или два значения: <action> или <action> <steps>")
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
        return _ManualCommand(kind="step", action=action, repeat_count=repeat_count)


def _apply_trajectory_smoothing(
    *,
    env,
    current_steps: list[dict[str, object]],
    smooth_anchor: int,
    episode_seed: int,
    command: _ManualCommand,
    html_path: Path,
    episode_idx: int,
    live_wind_cones: bool,
    live_poll_ms: int,
    live_generation: int,
    default_smooth_method: SmoothMethod,
    default_smooth_blend_fraction: float,
    default_smooth_alpha: float,
    default_smooth_window: int,
) -> tuple[list[dict[str, object]], int, float | None, np.ndarray, bool, bool, int]:
    # Сгладить участок траектории и пересчитать replay.
    step_count = len(current_steps)
    if step_count == 0:
        typer.echo("Нечего сглаживать: эпизод ещё не содержит шагов.")
        return (
            current_steps,
            smooth_anchor,
            None,
            np.empty(0, dtype=np.float32),
            False,
            False,
            live_generation,
        )

    if command.smooth_range_start is not None and command.smooth_range_end is not None:
        range_start_step = command.smooth_range_start
        range_end_step = command.smooth_range_end
        if range_start_step > range_end_step:
            range_start_step, range_end_step = range_end_step, range_start_step
        if range_start_step < 1 or range_end_step > step_count:
            typer.echo(
                f"Диапазон шагов должен быть в пределах 1–{step_count}, "
                f"получено {range_start_step}–{range_end_step}."
            )
            return (
                current_steps,
                smooth_anchor,
                None,
                np.empty(0, dtype=np.float32),
                False,
                False,
                live_generation,
            )
        range_start_idx = range_start_step - 1
        range_end_idx = range_end_step
        range_label = f"шаги {range_start_step}–{range_end_step}"
        update_anchor = False
    else:
        if step_count <= smooth_anchor:
            typer.echo("Нечего сглаживать: после последнего s ещё не было шагов.")
            return (
                current_steps,
                smooth_anchor,
                None,
                np.empty(0, dtype=np.float32),
                False,
                False,
                live_generation,
            )
        range_start_idx = smooth_anchor
        range_end_idx = step_count
        range_label = f"шаги {smooth_anchor + 1}–{step_count} (с последнего s)"
        update_anchor = True

    all_actions = [float(step["action"]) for step in current_steps]
    segment_actions = all_actions[range_start_idx:range_end_idx]
    if not segment_actions:
        typer.echo("Пустой диапазон для сглаживания.")
        return (
            current_steps,
            smooth_anchor,
            None,
            np.empty(0, dtype=np.float32),
            False,
            False,
            live_generation,
        )

    before_stats = count_action_transitions(segment_actions)
    blend_fraction = (
        command.smooth_blend_fraction
        if command.smooth_blend_fraction is not None
        else default_smooth_blend_fraction
    )
    smoothed_segment, smooth_stats = smooth_actions(
        segment_actions,
        method=command.smooth_method or default_smooth_method,
        blend_fraction=blend_fraction,
        alpha=command.smooth_alpha if command.smooth_alpha is not None else default_smooth_alpha,
        window=command.smooth_window or default_smooth_window,
        action_limit=float(env.action_limit),
    )
    changed_steps = sum(
        1
        for before, after in zip(segment_actions, smoothed_segment, strict=False)
        if abs(before - after) > 1e-6
    )
    mean_smoothed_action = float(np.mean(smoothed_segment)) if smoothed_segment else 0.0
    replay_actions = (
        all_actions[:range_start_idx]
        + smoothed_segment
        + all_actions[range_end_idx:]
    )

    typer.echo("")
    if before_stats.transition_count == 0:
        typer.echo(
            "⚠ Все action на участке одинаковы, угол, скорее всего, от ветра, "
            "а не от скачка action. Меняйте action на поворотах (например: "
            "8 15 -> -2 15), затем снова s."
        )
    typer.echo(
        f"Сглаживание {range_label} "
        f"({command.smooth_method or default_smooth_method}, "
        f"{len(segment_actions)} шагов, "
        f"переходов={before_stats.transition_count}, "
        f"макс. Δaction={before_stats.max_action_delta:.3f}, "
        f"коэф.={blend_fraction:.2f}, "
        f"рампа={smooth_stats.max_ramp_steps}/{smooth_stats.segment_steps} шагов, "
        f"средн. action={mean_smoothed_action:+.4f}, "
        f"изменено={changed_steps}) -> replay…"
    )

    replay_result = replay_episode_actions(
        env,
        seed=episode_seed,
        actions=replay_actions,
    )
    current_steps = replay_result.steps
    rewrite_env_current_trajectory(env, current_steps)
    new_anchor = len(current_steps) if update_anchor else smooth_anchor
    # Уникальный generation гарантирует, что live-viewer подхватит сглаженную траекторию
    # (file:// часто кэширует ping-pong слоты с тем же URL).
    live_generation = max(live_generation + 1, int(time.time() * 1000))
    _update_live_html(
        html_path=html_path,
        env=env,
        current_steps=current_steps,
        episode_idx=episode_idx,
        step_idx=max(len(current_steps) - 1, 0),
        live_wind_cones=live_wind_cones,
        live_poll_ms=live_poll_ms,
        generation=live_generation,
        smoothed=True,
    )
    typer.echo(
        f"Replay завершён: шагов={len(current_steps)} "
        f"total_reward={replay_result.total_reward:+.4f} "
        f"terminated={replay_result.terminated} truncated={replay_result.truncated}"
    )
    return (
        current_steps,
        new_anchor,
        replay_result.total_reward,
        replay_result.final_obs,
        replay_result.terminated,
        replay_result.truncated,
        live_generation,
    )


def _update_live_html(
    *,
    html_path: Path,
    env,
    current_steps: list[dict[str, object]],
    episode_idx: int,
    step_idx: int,
    live_wind_cones: bool,
    live_poll_ms: int,
    generation: int | None = None,
    smoothed: bool = False,
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
    step_label = f"шаг {step_idx + 1}"
    if smoothed:
        step_label = f"{step_label} · сглажено"
    save_live_trajectory_update(
        html_path,
        generation=generation if generation is not None else step_idx + 1,
        trajectory_traces=collect_trajectory_traces(
            env_idx=0,
            history=[],
            current_steps=current_steps,
            live_step_count=len(current_steps),
        ),
        title=(
            f"manual · эпизод {episode_idx + 1} · "
            f"{step_label}"
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
    smooth_method: SmoothMethod = typer.Option(
        "transition",
        "--smooth-method",
        help="Метод сглаживания: transition (весь участок с последнего s), ema, moving_average",
    ),
    smooth_blend_fraction: float = typer.Option(
        0.6,
        "--smooth-blend-fraction",
        min=0.05,
        max=1.0,
        help="Доля каждого платo action под рампу (0.6 по умолчанию)",
    ),
    smooth_alpha: float = typer.Option(
        0.25,
        "--smooth-alpha",
        min=0.01,
        max=1.0,
        help="Alpha для EMA-сглаживания",
    ),
    smooth_window: int = typer.Option(
        8,
        "--smooth-window",
        min=1,
        help="Окно для moving_average-сглаживания",
    ),
) -> None:
    # Интерактивный терминальный эпизод: человек выбирает action, среда пишет траекторию.
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
        save_live_trajectory_update(
            html_path,
            generation=0,
            trajectory_traces=[],
            title="manual · ожидание данных…",
            bounds=compute_trajectory_bounds([], world_bounds=env.world_bounds),
            wind_traces=[] if live_wind_cones else None,
            wind_key=0 if live_wind_cones else None,
            poll_interval_ms=live_poll_ms,
        )
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
            smooth_anchor = 0
            live_generation = 0
            current_steps: list[dict[str, object]] = []
            typer.echo("")
            typer.echo("=" * 80)
            typer.echo(f"Старт эпизода {episode_idx + 1}/{episodes}")
            typer.echo(f"Initial state: {env.render()}")
            typer.echo(f"Observation dim: {obs_vec.size}")
            typer.echo(
                "Команда s, с последнего s; s 500 1000, диапазон шагов; "
                "пересчитывает физику, ветер, reward и live HTML."
            )

            while True:
                command = _prompt_action_batch(
                    float(env.action_limit),
                    default_smooth_method=smooth_method,
                    default_smooth_blend_fraction=smooth_blend_fraction,
                    default_smooth_alpha=smooth_alpha,
                    default_smooth_window=smooth_window,
                )
                if command is None or command.kind == "quit":
                    typer.echo("Эпизод остановлен пользователем.")
                    return

                if command.kind == "smooth":
                    (
                        current_steps,
                        smooth_anchor,
                        replay_reward,
                        obs_vec,
                        terminated,
                        truncated,
                        live_generation,
                    ) = _apply_trajectory_smoothing(
                        env=env,
                        current_steps=current_steps,
                        smooth_anchor=smooth_anchor,
                        episode_seed=seed + episode_idx,
                        command=command,
                        html_path=html_path,
                        episode_idx=episode_idx,
                        live_wind_cones=live_wind_cones,
                        live_poll_ms=live_poll_ms,
                        live_generation=live_generation,
                        default_smooth_method=smooth_method,
                        default_smooth_blend_fraction=smooth_blend_fraction,
                        default_smooth_alpha=smooth_alpha,
                        default_smooth_window=smooth_window,
                    )
                    if replay_reward is not None:
                        episode_reward = replay_reward
                    step_idx = len(current_steps)
                    if terminated or truncated:
                        typer.echo(
                            f"Эпизод завершён после smooth: terminated={terminated} "
                            f"truncated={truncated} steps={step_idx} "
                            f"total_reward={episode_reward:+.4f}"
                        )
                        break
                    continue

                action = command.action
                repeat_count = command.repeat_count

                for repeat_idx in range(repeat_count):
                    next_obs, reward_value, terminated, truncated, info = env.step(
                        np.asarray([action], dtype=np.float32)
                    )
                    step_record = env.consume_step_record()
                    if step_record:
                        current_steps.append(step_record)
                        if repeat_count == 1:
                            live_generation += 1
                            _update_live_html(
                                html_path=html_path,
                                env=env,
                                current_steps=current_steps,
                                episode_idx=episode_idx,
                                step_idx=step_idx,
                                live_wind_cones=live_wind_cones,
                                live_poll_ms=live_poll_ms,
                                generation=live_generation,
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
                    live_generation += 1
                    _update_live_html(
                        html_path=html_path,
                        env=env,
                        current_steps=current_steps,
                        episode_idx=episode_idx,
                        step_idx=step_idx - 1,
                        live_wind_cones=live_wind_cones,
                        live_poll_ms=live_poll_ms,
                        generation=live_generation,
                    )
                if terminated or truncated:
                    break

            if episode_idx + 1 < episodes and not typer.confirm("Начать следующий эпизод?"):
                break
    except KeyboardInterrupt:
        typer.echo("\nПрервано пользователем.")
    finally:
        env.close()
