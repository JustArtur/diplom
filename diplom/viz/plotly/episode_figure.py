"""Сборка Plotly-фигур эпизодов: общая логика для rollout, live-render и placeholder."""

from __future__ import annotations

from dataclasses import dataclass
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

# Live-рендер траекторий.
TRAJECTORY_LIVE_WIND_OVERLAY: dict[str, Any] = {
    **WIND_OVERLAY_DEFAULTS,
    "stride_lon": WIND_OVERLAY_DEFAULTS["stride_lon"] * 2,
    "stride_lat": WIND_OVERLAY_DEFAULTS["stride_lat"] * 2,
    "cone_scale": 0.6,
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


def wind_overlay_cache_key(sim_time: np.datetime64) -> int:
    """Ключ кэша слоя ветра: один слой конусов на каждый час ERA5."""
    return int(np.datetime64(sim_time, "h").astype(np.int64))


@dataclass(frozen=True, slots=True)
class LiveTrainingFigureParts:
    trajectory_traces: list
    title: str
    bounds: TrajectoryBounds
    wind_traces: list | None = None
    wind_key: int | None = None


def collect_trajectory_traces(
    *,
    env_idx: int,
    history: list[EpisodeVizData],
    current_steps: list[dict[str, Any]],
    live_step_count: int,
    line_width: int = 7,
) -> list:
    traces: list = []
    for ep in history:
        traces.extend(build_episode_traces(ep, line_width=line_width))
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
        traces.extend(build_episode_traces(live_ep, line_width=line_width))
    return traces


def build_live_training_parts(
    *,
    env_idx: int,
    history: list[EpisodeVizData],
    current_steps: list[dict[str, Any]],
    live_step_count: int,
    bounds: TrajectoryBounds,
    num_timesteps: int,
    episode_count: int,
    wind_traces: list | None = None,
    wind_key: int | None = None,
) -> LiveTrainingFigureParts:
    title = (
        f"env_{env_idx:03d} · "
        f"шаг {num_timesteps:,} · "
        f"завершено эпизодов: {episode_count}"
    )
    return LiveTrainingFigureParts(
        trajectory_traces=collect_trajectory_traces(
            env_idx=env_idx,
            history=history,
            current_steps=current_steps,
            live_step_count=live_step_count,
        ),
        title=title,
        bounds=bounds,
        wind_traces=wind_traces,
        wind_key=wind_key,
    )


def build_placeholder_live_figure(
    env_idx: int,
    world_bounds: WorldBounds | None,
) -> go.Figure:
    bounds = compute_trajectory_bounds([], world_bounds=world_bounds)
    fig = go.Figure()
    apply_figure_layout(fig, f"env_{env_idx:03d} · ожидание данных…", bounds)
    return fig
