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

from diplom.shared_constants import MAX_HEIGHT, WORLD_SIZE

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
    def xy_span(self) -> float:
        return max(self.xmax - self.xmin, self.ymax - self.ymin)

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
) -> TrajectoryBounds:
    """Вычислить границы bbox по всем позициям траекторий + целевым точкам.

    Args:
        episodes: список эпизодов с шагами.
        extra_steps: дополнительные шаги (текущий незавершённый эпизод).
        margin: относительный отступ от краёв bbox (0.25 = 25% от span).
        min_xy_span: минимальный размах по XY (м), чтобы не схлопываться в точку.
        min_z_span: минимальный размах по Z (м).
    """
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
            xmin=0, xmax=WORLD_SIZE,
            ymin=0, ymax=WORLD_SIZE,
            zmin=0, zmax=MAX_HEIGHT,
        )

    pos = np.array(all_pos, dtype=np.float32)
    xmin, ymin, zmin = pos.min(axis=0)
    xmax, ymax, zmax = pos.max(axis=0)

    # Применяем минимальный размах
    cx, cy, cz = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
    xy_half = max((xmax - xmin) / 2, (ymax - ymin) / 2, min_xy_span / 2)
    z_half = max((zmax - zmin) / 2, min_z_span / 2)

    xmin, xmax = cx - xy_half, cx + xy_half
    ymin, ymax = cy - xy_half, cy + xy_half
    zmin, zmax = max(0.0, cz - z_half), cz + z_half

    # Добавляем отступ
    xmin -= xy_half * margin
    xmax += xy_half * margin
    ymin -= xy_half * margin
    ymax += xy_half * margin
    zmin = max(0.0, zmin - z_half * margin)
    zmax += z_half * margin

    # Зажимаем в допустимые пределы мира
    xmin = max(0.0, xmin)
    xmax = min(WORLD_SIZE, xmax)
    ymin = max(0.0, ymin)
    ymax = min(WORLD_SIZE, ymax)
    zmin = max(0.0, zmin)
    zmax = min(MAX_HEIGHT, zmax)

    return TrajectoryBounds(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                            zmin=zmin, zmax=zmax)


def build_figure(
    episodes: List[EpisodeVizData],
    title: str = "Траектория полёта аэростата",
    bounds: Optional[TrajectoryBounds] = None,
    extra_steps: Optional[List[dict]] = None,
) -> go.Figure:
    """Построить интерактивный 3D-график траекторий.

    Args:
        episodes: список эпизодов.
        title: заголовок.
        bounds: bbox для масштаба осей; вычисляется автоматически если None.
        extra_steps: шаги текущего незавершённого эпизода (для bounds).
    """
    if bounds is None:
        bounds = compute_trajectory_bounds(episodes, extra_steps)

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

    hover_texts = [
        (
            f"шаг {i}<br>"
            f"x={x[i]:.0f} м, y={y[i]:.0f} м, z={z[i]:.0f} м<br>"
            f"reward={episode.steps[i].get('reward', 0.0):.3f}<br>"
            f"действие={episode.steps[i].get('action', 0.0):.3f}<br>"
            f"dist={episode.steps[i].get('distance_to_target', 0.0):.1f} м"
        )
        for i in range(n)
    ]

    progress = np.linspace(0, 1, n).tolist()
    line_colorscale = [[0.0, "rgba(180,180,180,0.35)"], [1.0, color]]

    traces: List[go.BaseTraceType] = [
        go.Scatter3d(
            x=x.tolist(), y=y.tolist(), z=z.tolist(),
            mode="lines",
            name=name,
            line=dict(color=progress, colorscale=line_colorscale, width=4),
            hovertemplate="%{text}<extra></extra>",
            text=hover_texts,
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

    Если bounds передан — используем его для вертикального масштаба, но по X/Y
    всегда показываем весь мир (0..WORLD_SIZE), чтобы ни одна траектория и цель
    не оказывалась «за кадром».
    """

    if bounds is not None:
        # По горизонтали всегда показываем весь мир — так гарантированно видны
        # любые траектории и цели, даже если они далеко от текущего bbox.
        x_range = [0.0, WORLD_SIZE]
        y_range = [0.0, WORLD_SIZE]
        # Z: всегда весь допустимый диапазон высот, чтобы цели на больших
        # высотах (например, 10–20 км) не обрезались.
        z_range = [0.0, MAX_HEIGHT]

        # Соотношение сторон: x=y (квадратные горизонтальные оси), z пропорционально
        xy_span = float(WORLD_SIZE)
        z_span = z_range[1] - z_range[0]
        # Вертикальный масштаб: минимум 0.4 от горизонтального, максимум 1.0
        z_ratio = float(np.clip(z_span / xy_span, 0.4, 1.0))

        def _ticks(lo: float, hi: float, n: int = 6) -> tuple[list, list]:
            vals = np.linspace(lo, hi, n).tolist()
            labels = [f"{v:.0f}" for v in vals]
            return vals, labels

        x_ticks, x_labels = _ticks(*x_range)
        y_ticks, y_labels = _ticks(*y_range)
        z_ticks, z_labels = _ticks(*z_range)
    else:
        x_range = [0, WORLD_SIZE]
        y_range = [0, WORLD_SIZE]
        z_range = [0, MAX_HEIGHT]
        z_ratio = 0.3
        x_ticks = np.linspace(0, WORLD_SIZE, 6).tolist()
        x_labels = [f"{v/1000:.0f} km" for v in x_ticks]
        y_ticks, y_labels = x_ticks, x_labels
        z_ticks = np.linspace(0, MAX_HEIGHT, 6).tolist()
        z_labels = [f"{v:.0f}" for v in z_ticks]

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
            aspectratio=dict(x=1.0, y=1.0, z=z_ratio),
            bgcolor="rgba(10,10,30,1)",
        ),
        legend=dict(groupclick="toggleitem", bgcolor="rgba(30,30,30,0.8)"),
        margin=dict(l=0, r=0, b=0, t=50),
        template="plotly_dark",
        paper_bgcolor="rgba(15,15,25,1)",
    )


# ──────────────────── Утилиты ────────────────────


