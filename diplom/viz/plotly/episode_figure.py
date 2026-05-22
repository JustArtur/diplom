"""Сборка Plotly-фигур эпизодов: общая логика для rollout, live-render и placeholder."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go

from diplom.viz.plotly.trajectory import (
    EpisodeVizData,
    TrajectoryBounds,
    apply_figure_layout,
    build_episode_traces,
    build_figure,
    compute_trajectory_bounds,
    save_figure,
)
from diplom.wind.interp import WindInterpolator
from diplom.world import WorldBounds

WIND_OVERLAY_DEFAULTS: dict[str, Any] = {
    "stride_lon": 2,
    "stride_lat": 2,
    "stride_altitude_m": 800.0,
    "cone_sizeref": 25.0,
    "altitude_unit": "m",
    "show_colorbar": False,
}


def latest_sim_time(
    current_steps: list[dict[str, Any]],
    history: list[EpisodeVizData],
) -> np.datetime64 | None:
    if current_steps:
        return np.datetime64(current_steps[-1]["sim_time"])
    if history and history[-1].steps:
        return np.datetime64(history[-1].steps[-1]["sim_time"])
    return None


def build_wind_overlay_traces(
    interpolator: WindInterpolator,
    sim_time: np.datetime64,
    **kwargs: Any,
) -> list:
    from diplom.viz.plotly.wind import resolve_wind_time
    from diplom.viz.plotly.wind_traces import build_wind_cone_traces

    overlay = {**WIND_OVERLAY_DEFAULTS, **kwargs}
    plot_time = resolve_wind_time(interpolator, sim_time)
    return build_wind_cone_traces(interpolator, plot_time, **overlay)


def rollout_results_to_episodes(results: list[Any]) -> list[EpisodeVizData]:
    episodes: list[EpisodeVizData] = []
    for idx, result in enumerate(results):
        episodes.append(
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
        )
    return episodes


def build_rollout_figure(
    episodes: list[EpisodeVizData],
    *,
    title: str,
    wind_interpolator: WindInterpolator,
) -> go.Figure:
    bounds = compute_trajectory_bounds(episodes, world_bounds=wind_interpolator.world_bounds)
    wind_traces: list = []
    if episodes and episodes[0].steps:
        sim_time = np.datetime64(episodes[0].steps[-1]["sim_time"])
        wind_traces = build_wind_overlay_traces(wind_interpolator, sim_time)
    return build_figure(
        episodes=episodes,
        title=title,
        bounds=bounds,
        wind_traces=wind_traces,
    )


def save_rollout_figure(
    episodes: list[EpisodeVizData],
    *,
    title: str,
    wind_interpolator: WindInterpolator,
    output_path: Path,
) -> None:
    fig = build_rollout_figure(
        episodes,
        title=title,
        wind_interpolator=wind_interpolator,
    )
    save_figure(fig, output_path)


def build_live_training_figure(
    *,
    env_idx: int,
    history: list[EpisodeVizData],
    current_steps: list[dict[str, Any]],
    live_step_count: int,
    bounds: TrajectoryBounds,
    num_timesteps: int,
    episode_count: int,
    wind_dataset_path: Path | None = None,
    get_wind_interpolator: Callable[[Path], WindInterpolator] | None = None,
) -> go.Figure:
    fig = go.Figure()

    if wind_dataset_path is not None and get_wind_interpolator is not None:
        sim_time = latest_sim_time(current_steps, history)
        if sim_time is not None:
            interpolator = get_wind_interpolator(wind_dataset_path)
            for trace in build_wind_overlay_traces(interpolator, sim_time):
                fig.add_trace(trace)

    for ep in history:
        for trace in build_episode_traces(ep):
            fig.add_trace(trace)

    if current_steps:
        fallback_target = history[-1].target_position.tolist() if history else [0.0, 0.0, 0.0]
        live_target = np.array(
            current_steps[-1].get("target_position", fallback_target),
            dtype=np.float32,
        )
        live_ep = EpisodeVizData(
            env_idx=env_idx,
            steps=current_steps,
            target_position=live_target,
            label=f"сейчас ({live_step_count} шагов)",
        )
        for trace in build_episode_traces(live_ep):
            fig.add_trace(trace)

    title = (
        f"env_{env_idx:03d} · "
        f"шаг {num_timesteps:,} · "
        f"завершено эпизодов: {episode_count}"
    )
    apply_figure_layout(fig, title, bounds)
    return fig


def build_placeholder_live_figure(
    env_idx: int,
    world_bounds: WorldBounds | None,
) -> go.Figure:
    bounds = compute_trajectory_bounds([], world_bounds=world_bounds)
    fig = go.Figure()
    apply_figure_layout(fig, f"env_{env_idx:03d} · ожидание данных…", bounds)
    return fig
