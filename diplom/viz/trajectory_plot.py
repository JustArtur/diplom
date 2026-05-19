"""Интерактивная 3D-визуализация траектории полёта аэростата (Plotly HTML).

Публичный API:
  compute_trajectory_bounds(episodes, extra_steps, margin) → TrajectoryBounds
  build_figure(episodes, title, bounds)                    → go.Figure
  build_episode_traces(episode)                            → List[go.BaseTraceType]
  apply_figure_layout(fig, title, bounds)
  save_figure(fig, path)                                   → standalone HTML
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import plotly.graph_objects as go

from diplom.shared_constants import MAX_HEIGHT
from diplom.world import WorldBounds

# Цветовая палитра траекторий разных сред.
_TRAJ_PALETTE: List[str] = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _env_color(env_idx: int) -> str:
    return _TRAJ_PALETTE[env_idx % len(_TRAJ_PALETTE)]


# ──────────────────── Типы данных ────────────────────

@dataclass
class EpisodeVizData:
    """Данные одного эпизода для построения графика."""

    env_idx: int
    steps: List[dict]       # каждый шаг: position, wind, action, reward, sim_time, ...
    target_position: np.ndarray
    label: str = ""         # отображаемое имя в легенде


# ──────────────────── TrajectoryBounds ────────────────────

@dataclass
class TrajectoryBounds:
    """Ограничивающий параллелепипед траекторий — используется для масштаба осей."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    @property
    def x_span(self) -> float:
        return self.xmax - self.xmin

    @property
    def y_span(self) -> float:
        return self.ymax - self.ymin

    @property
    def xy_span(self) -> float:
        return max(self.x_span, self.y_span)

    @property
    def z_span(self) -> float:
        return self.zmax - self.zmin

    @property
    def center(self) -> tuple[float, float, float]:
        return (
            (self.xmin + self.xmax) / 2,
            (self.ymin + self.ymax) / 2,
            (self.zmin + self.zmax) / 2,
        )


def compute_trajectory_bounds(
    episodes: List[EpisodeVizData],
    extra_steps: Optional[List[dict]] = None,
    margin: float = 0.25,
    min_xy_span: float = 1000.0,
    min_z_span: float = 200.0,
    world_bounds: Optional[WorldBounds] = None,
) -> TrajectoryBounds:
    """Вычислить границы bbox по всем позициям траекторий + целевым точкам.

    Args:
        episodes: список эпизодов с шагами.
        extra_steps: дополнительные шаги (текущий незавершённый эпизод).
        margin: относительный отступ от краёв bbox (0.25 = 25% от span).
        min_xy_span: минимальный размах по XY (м), чтобы не схлопываться в точку.
        min_z_span: минимальный размах по Z (м).
        world_bounds: реальные границы мира, если график нужно синхронизировать по датасету.
    """
    if world_bounds is not None:
        return _bounds_from_world(world_bounds)

    all_pos: List[List[float]] = []

    for ep in episodes:
        for step in ep.steps:
            all_pos.append(step["position"])
        all_pos.append(ep.target_position.tolist())

    if extra_steps:
        for step in extra_steps:
            all_pos.append(step["position"])
        if extra_steps:
            tp = extra_steps[-1].get("target_position")
            if tp:
                all_pos.append(tp)

    if not all_pos:
        return TrajectoryBounds(
            xmin=0.0, xmax=1.0,
            ymin=0.0, ymax=1.0,
            zmin=0.0, zmax=MAX_HEIGHT,
        )

    return _bounds_from_positions(
        all_pos,
        margin=margin,
        min_xy_span=min_xy_span,
        min_z_span=min_z_span,
        world_bounds=None,
    )


def compute_trajectory_bounds_from_positions(
    positions: List[List[float]],
    *,
    margin: float = 0.25,
    min_xy_span: float = 1000.0,
    min_z_span: float = 200.0,
    world_bounds: Optional[WorldBounds] = None,
) -> TrajectoryBounds:
    """Вычислить bbox только по списку координат (без загрузки полных шагов)."""
    if world_bounds is not None:
        return _bounds_from_world(world_bounds)
    return _bounds_from_positions(
        positions,
        margin=margin,
        min_xy_span=min_xy_span,
        min_z_span=min_z_span,
        world_bounds=None,
    )


def compute_trajectory_bounds_from_extents(
    min_xyz: np.ndarray | None,
    max_xyz: np.ndarray | None,
    *,
    margin: float = 0.25,
    min_xy_span: float = 1000.0,
    min_z_span: float = 200.0,
    world_bounds: Optional[WorldBounds] = None,
) -> TrajectoryBounds:
    """BBox по уже посчитанным min/max координат (без списка всех точек)."""
    if world_bounds is not None:
        return _bounds_from_world(world_bounds)
    if min_xyz is None or max_xyz is None:
        return _bounds_from_extent_values(
            xmin_raw=0.0, ymin_raw=0.0, zmin=0.0,
            xmax_raw=1.0, ymax_raw=1.0, zmax=MAX_HEIGHT,
            margin=margin, min_xy_span=min_xy_span, min_z_span=min_z_span,
            has_points=False,
        )
    return _bounds_from_extent_values(
        xmin_raw=float(min_xyz[0]), ymin_raw=float(min_xyz[1]), zmin=float(min_xyz[2]),
        xmax_raw=float(max_xyz[0]), ymax_raw=float(max_xyz[1]), zmax=float(max_xyz[2]),
        margin=margin, min_xy_span=min_xy_span, min_z_span=min_z_span,
        has_points=True,
    )


def _bounds_from_world(world_bounds: WorldBounds) -> TrajectoryBounds:
    return TrajectoryBounds(
        xmin=world_bounds.x_min,
        xmax=world_bounds.x_max,
        ymin=world_bounds.y_min,
        ymax=world_bounds.y_max,
        zmin=world_bounds.z_min,
        zmax=world_bounds.z_max,
    )


def _bounds_from_positions(
    all_pos: List[List[float]],
    *,
    margin: float,
    min_xy_span: float,
    min_z_span: float,
    world_bounds: Optional[WorldBounds] = None,
) -> TrajectoryBounds:
    if not all_pos:
        if world_bounds is not None:
            return _bounds_from_world(world_bounds)
        return TrajectoryBounds(
            xmin=0.0, xmax=1.0,
            ymin=0.0, ymax=1.0,
            zmin=0.0, zmax=MAX_HEIGHT,
        )

    pos = np.array(all_pos, dtype=np.float32)
    xmin_raw, ymin_raw, zmin = pos.min(axis=0)
    xmax_raw, ymax_raw, zmax = pos.max(axis=0)
    return _bounds_from_extent_values(
        xmin_raw=float(xmin_raw), ymin_raw=float(ymin_raw), zmin=float(zmin),
        xmax_raw=float(xmax_raw), ymax_raw=float(ymax_raw), zmax=float(zmax),
        margin=margin, min_xy_span=min_xy_span, min_z_span=min_z_span,
        has_points=True,
    )


def _bounds_from_extent_values(
    *,
    xmin_raw: float,
    ymin_raw: float,
    zmin: float,
    xmax_raw: float,
    ymax_raw: float,
    zmax: float,
    margin: float,
    min_xy_span: float,
    min_z_span: float,
    has_points: bool,
) -> TrajectoryBounds:
    if not has_points:
        return TrajectoryBounds(
            xmin=0.0, xmax=1.0,
            ymin=0.0, ymax=1.0,
            zmin=0.0, zmax=MAX_HEIGHT,
        )

    cx = (xmin_raw + xmax_raw) / 2
    cy = (ymin_raw + ymax_raw) / 2
    x_half = max((xmax_raw - xmin_raw) / 2, min_xy_span / 2)
    y_half = max((ymax_raw - ymin_raw) / 2, min_xy_span / 2)
    xmin, xmax = cx - x_half, cx + x_half
    ymin, ymax = cy - y_half, cy + y_half

    xmin -= x_half * margin
    xmax += x_half * margin
    ymin -= y_half * margin
    ymax += y_half * margin

    cz = (zmin + zmax) / 2
    z_half = max((zmax - zmin) / 2, min_z_span / 2)
    zmin = max(0.0, cz - z_half)
    zmax = cz + z_half

    zmin = max(0.0, zmin - z_half * margin)
    zmax = min(MAX_HEIGHT, zmax + z_half * margin)

    return TrajectoryBounds(
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax,
    )


def build_figure(
    episodes: List[EpisodeVizData],
    title: str = "Траектория полёта аэростата",
    bounds: Optional[TrajectoryBounds] = None,
    extra_steps: Optional[List[dict]] = None,
    world_bounds: Optional[WorldBounds] = None,
) -> go.Figure:
    """Построить интерактивный 3D-график траекторий.

    Args:
        episodes: список эпизодов.
        title: заголовок.
        bounds: bbox для масштаба осей; вычисляется автоматически если None.
        extra_steps: шаги текущего незавершённого эпизода (для bounds).
        world_bounds: реальные границы мира, если график нужно синхронизировать по датасету.
    """
    if bounds is None:
        bounds = compute_trajectory_bounds(episodes, extra_steps, world_bounds=world_bounds)

    fig = go.Figure()

    for ep in episodes:
        for trace in build_episode_traces(ep):
            fig.add_trace(trace)

    apply_figure_layout(fig, title, bounds)
    return fig


def save_figure(fig: go.Figure, path: Path) -> None:
    """Сохранить фигуру как standalone HTML (CDN Plotly, не встроен в файл)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")


# ──────────────────── Построение траектории ────────────────────

def build_episode_traces(episode: EpisodeVizData) -> List[go.BaseTraceType]:
    if not episode.steps:
        return []

    color = _env_color(episode.env_idx)
    positions = np.array([s["position"] for s in episode.steps], dtype=np.float32)
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    n = len(x)

    name = episode.label or f"env_{episode.env_idx}"
    group = f"ep_{episode.env_idx}_{id(episode)}"

    rewards = np.fromiter(
        (float(s.get("reward", 0.0)) for s in episode.steps), dtype=np.float32, count=n,
    )
    actions = np.fromiter(
        (float(s.get("action", 0.0)) for s in episode.steps), dtype=np.float32, count=n,
    )
    distances = np.fromiter(
        (float(s.get("distance_to_target", 0.0)) for s in episode.steps), dtype=np.float32, count=n,
    )
    customdata = np.column_stack((rewards, actions, distances))

    progress = np.linspace(0, 1, n, dtype=np.float32)
    line_colorscale = [[0.0, "rgba(180,180,180,0.35)"], [1.0, color]]

    traces: List[go.BaseTraceType] = [
        go.Scatter3d(
            x=x, y=y, z=z,
            mode="lines",
            name=name,
            line=dict(color=progress, colorscale=line_colorscale, width=4),
            customdata=customdata,
            hovertemplate=(
                "шаг %{pointNumber}<br>"
                "x=%{x:.0f} м, y=%{y:.0f} м, z=%{z:.0f} м<br>"
                "reward=%{customdata[0]:.3f}<br>"
                "действие=%{customdata[1]:.3f}<br>"
                "dist=%{customdata[2]:.1f} м"
                "<extra></extra>"
            ),
            legendgroup=group,
        ),
        go.Scatter3d(
            x=[x[0]], y=[y[0]], z=[z[0]],
            mode="markers",
            name=f"{name} · старт",
            marker=dict(color="lime", size=8, symbol="diamond",
                        line=dict(color="black", width=1)),
            hovertemplate=(
                f"{name} — старт<br>x={x[0]:.0f} м, y={y[0]:.0f} м, z={z[0]:.0f} м"
                "<extra></extra>"
            ),
            legendgroup=group,
            showlegend=False,
        ),
        go.Scatter3d(
            x=[episode.target_position[0]],
            y=[episode.target_position[1]],
            z=[episode.target_position[2]],
            mode="markers",
            name=f"{name} · цель",
            marker=dict(color="red", size=12, symbol="x"),
            hovertemplate=(
                f"{name} — цель<br>"
                f"x={episode.target_position[0]:.0f} м, "
                f"y={episode.target_position[1]:.0f} м, "
                f"z={episode.target_position[2]:.0f} м"
                "<extra></extra>"
            ),
            legendgroup=group,
            showlegend=False,
        ),
        go.Scatter3d(
            x=[x[-1]], y=[y[-1]], z=[z[-1]],
            mode="markers",
            name=f"{name} · конец",
            marker=dict(
                color="gold" if episode.steps[-1].get("terminated") else "silver",
                size=7, symbol="circle",
                line=dict(color="black", width=1),
            ),
            hovertemplate=(
                f"{name} — {'успех ✓' if episode.steps[-1].get('terminated') else 'truncated'}<br>"
                f"x={x[-1]:.0f} м, y={y[-1]:.0f} м, z={z[-1]:.0f} м"
                "<extra></extra>"
            ),
            legendgroup=group,
            showlegend=False,
        ),
    ]
    return traces


# ──────────────────── Layout ────────────────────

def apply_figure_layout(
    fig: go.Figure,
    title: str,
    bounds: Optional[TrajectoryBounds] = None,
) -> None:
    """Применить layout с осями и камерой.

    Если bounds передан — используем его для всех осей. Для X/Y это означает
    реальные границы мира, для Z — границы данных траекторий.
    """

    if bounds is not None:
        x_range = [bounds.xmin, bounds.xmax]
        y_range = [bounds.ymin, bounds.ymax]
        z_range = [bounds.zmin, bounds.zmax]

        horizontal_span = max(bounds.x_span, bounds.y_span, 1.0)
        z_ratio = float(np.clip(max(bounds.z_span, 1.0) / horizontal_span, 0.4, 1.0))

        def _ticks(lo: float, hi: float, n: int = 6) -> tuple[list, list]:
            vals = np.linspace(lo, hi, n).tolist()
            labels = [f"{v/1000:.0f} km" if abs(hi - lo) >= 10_000 else f"{v:.0f}" for v in vals]
            return vals, labels

        x_ticks, x_labels = _ticks(*x_range)
        y_ticks, y_labels = _ticks(*y_range)
        z_ticks, z_labels = _ticks(*z_range)
    else:
        x_range = [0.0, 1.0]
        y_range = [0.0, 1.0]
        z_range = [0.0, MAX_HEIGHT]
        z_ratio = 0.3
        x_ticks = np.linspace(0.0, 1.0, 6).tolist()
        x_labels = [f"{v:.0f}" for v in x_ticks]
        y_ticks, y_labels = x_ticks, x_labels
        z_ticks = np.linspace(0, MAX_HEIGHT, 6).tolist()
        z_labels = [f"{v:.0f}" for v in z_ticks]

    x_span = max(bounds.x_span, 1.0) if bounds is not None else 1.0
    y_span = max(bounds.y_span, 1.0) if bounds is not None else 1.0

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=15)),
        scene=dict(
            xaxis=dict(
                title="X, м", range=x_range,
                tickvals=x_ticks, ticktext=x_labels,
            ),
            yaxis=dict(
                title="Y, м", range=y_range,
                tickvals=y_ticks, ticktext=y_labels,
            ),
            zaxis=dict(
                title="Высота, м", range=z_range,
                tickvals=z_ticks, ticktext=z_labels,
            ),
            aspectmode="manual",
            aspectratio=dict(
                x=x_span / max(x_span, y_span),
                y=y_span / max(x_span, y_span),
                z=z_ratio,
            ),
            bgcolor="rgba(10,10,30,1)",
        ),
        legend=dict(groupclick="toggleitem", bgcolor="rgba(30,30,30,0.8)"),
        margin=dict(l=0, r=0, b=0, t=50),
        template="plotly_dark",
        paper_bgcolor="rgba(15,15,25,1)",
    )


# ──────────────────── Утилиты ────────────────────


