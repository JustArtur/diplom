from __future__ import annotations

import queue
import traceback
from dataclasses import dataclass
from multiprocessing import Queue
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from diplom.world import WorldBounds
from diplom.viz.trajectory_plot import (
    EpisodeVizData,
    TrajectoryBounds,
    apply_figure_layout,
    build_episode_traces,
    compute_trajectory_bounds,
    save_figure,
)


STOP_SENTINEL = "__STOP__"


@dataclass(frozen=True, slots=True)
class TrajectoryRenderRequest:
    num_timesteps: int
    n_envs: int
    episode_counts: dict[int, int]
    history: dict[int, list[dict[str, Any]]]
    current_steps: dict[int, list[dict[str, Any]]]
    world_bounds: WorldBounds | None = None


def start_trajectory_render_worker(
    *,
    ctx,
    output_dir: Path,
) -> tuple[Queue, Any]:
    """Создать очередь и отдельный процесс для рендера траекторий."""
    task_queue: Queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_render_worker_main,
        args=(task_queue, output_dir),
        daemon=True,
    )
    process.start()
    return task_queue, process


def stop_trajectory_render_worker(task_queue: Queue | None, process: Any | None) -> None:
    """Остановить процесс рендера, если он был запущен."""
    if task_queue is None or process is None:
        return

    try:
        task_queue.put_nowait(STOP_SENTINEL)
    except queue.Full:
        try:
            task_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            task_queue.put_nowait(STOP_SENTINEL)
        except queue.Full:
            pass

    process.join(timeout=5.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5.0)


def submit_trajectory_render(task_queue: Queue | None, request: TrajectoryRenderRequest) -> None:
    """Передать последний снапшот траекторий в очередь рендера."""
    if task_queue is None:
        return

    payload = {
        "num_timesteps": request.num_timesteps,
        "n_envs": request.n_envs,
        "episode_counts": request.episode_counts,
        "history": request.history,
        "current_steps": request.current_steps,
        "world_bounds": request.world_bounds,
    }

    try:
        task_queue.put_nowait(payload)
    except queue.Full:
        try:
            task_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            task_queue.put_nowait(payload)
        except queue.Full:
            pass


def _render_worker_main(task_queue: Queue, output_dir: Path) -> None:
    """Фоновый воркер, который строит HTML-файлы траекторий."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        while True:
            task = task_queue.get()
            if task == STOP_SENTINEL:
                break

            try:
                request = _decode_request(task)
                _render_snapshot(request, output_dir)
            except Exception:  # noqa: BLE001
                traceback.print_exc()
    except Exception:  # noqa: BLE001
        traceback.print_exc()


def _decode_request(task: Mapping[str, Any]) -> TrajectoryRenderRequest:
    return TrajectoryRenderRequest(
        num_timesteps=int(task["num_timesteps"]),
        n_envs=int(task["n_envs"]),
        episode_counts={int(k): int(v) for k, v in task["episode_counts"].items()},
        history={
            int(env_idx): [dict(step) for step in episodes]
            for env_idx, episodes in task["history"].items()
        },
        current_steps={
            int(env_idx): [dict(step) for step in steps]
            for env_idx, steps in task["current_steps"].items()
        },
        world_bounds=task.get("world_bounds"),
    )


def _episode_from_payload(env_idx: int, episode: Mapping[str, Any]) -> EpisodeVizData:
    return EpisodeVizData(
        env_idx=env_idx,
        steps=[dict(step) for step in episode["steps"]],
        target_position=np.asarray(episode["target_position"], dtype=np.float32),
        label=str(episode.get("label", "")),
    )


def _render_snapshot(
    request: TrajectoryRenderRequest,
    output_dir: Path,
) -> None:
    """Построить и сохранить HTML для снимка траекторий."""
    history: dict[int, list[EpisodeVizData]] = {}
    for env_idx, episodes in request.history.items():
        history[env_idx] = [_episode_from_payload(env_idx, episode) for episode in episodes]

    current_steps: dict[int, list[dict[str, Any]]] = {
        env_idx: [dict(step) for step in steps]
        for env_idx, steps in request.current_steps.items()
    }

    all_episodes = [ep for env_hist in history.values() for ep in env_hist]
    all_current = [step for steps in current_steps.values() for step in steps]
    bounds = compute_trajectory_bounds(all_episodes, all_current, world_bounds=request.world_bounds)

    for env_idx in range(request.n_envs):
        history_items = history.get(env_idx, [])
        current_env_steps = current_steps.get(env_idx, [])
        if not history_items and not current_env_steps:
            continue

        fig = _build_figure_for_env(
            env_idx=env_idx,
            history=history_items,
            current_steps=current_env_steps,
            bounds=bounds,
            num_timesteps=request.num_timesteps,
            episode_count=request.episode_counts.get(env_idx, 0),
        )
        save_figure(fig, output_dir / f"env_{env_idx:03d}.html")


def _build_figure_for_env(
    *,
    env_idx: int,
    history: list[EpisodeVizData],
    current_steps: list[dict[str, Any]],
    bounds: TrajectoryBounds,
    num_timesteps: int,
    episode_count: int,
):
    import plotly.graph_objects as go

    from diplom.viz.trajectory_plot import build_episode_traces

    fig = go.Figure()
    for ep in history:
        for trace in build_episode_traces(ep):
            fig.add_trace(trace)

    if current_steps:
        fallback_target = history[-1].target_position.tolist() if history else [0.0, 0.0, 0.0]
        live_target = np.array(current_steps[-1].get("target_position", fallback_target), dtype=np.float32)
        live_ep = EpisodeVizData(
            env_idx=env_idx,
            steps=current_steps,
            target_position=live_target,
            label=f"сейчас ({len(current_steps)} шагов)",
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
