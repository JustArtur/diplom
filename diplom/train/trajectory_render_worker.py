from __future__ import annotations

import pickle
import queue
import traceback
from dataclasses import dataclass
from multiprocessing import Queue
from pathlib import Path
from typing import Any

import numpy as np

from diplom.world import WorldBounds
from diplom.viz.trajectory_plot import (
    EpisodeVizData,
    TrajectoryBounds,
    apply_figure_layout,
    build_episode_traces,
    compute_trajectory_bounds_from_extents,
    save_figure,
)
from diplom.train.trajectory_steps_io import (
    EpisodeFileRef,
    accumulate_position_extents,
    include_position_in_extents,
    load_last_target_from_jsonl,
    load_viz_steps_jsonl,
)


STOP_SENTINEL = "__STOP__"
SNAPSHOTS_SUBDIR = "_snapshots"
SNAPSHOT_FILENAME = "snapshot_{num_timesteps:012d}.pkl"


@dataclass(frozen=True, slots=True)
class TrajectoryRenderRequest:
    num_timesteps: int
    n_envs: int
    episode_counts: dict[int, int]
    history: dict[int, list[EpisodeFileRef]]
    current_steps_paths: dict[int, Path]
    current_step_counts: dict[int, int]
    world_bounds: WorldBounds | None = None


def snapshots_dir(output_dir: Path) -> Path:
    return Path(output_dir) / SNAPSHOTS_SUBDIR


def snapshot_path_for(output_dir: Path, num_timesteps: int) -> Path:
    return snapshots_dir(output_dir) / SNAPSHOT_FILENAME.format(num_timesteps=num_timesteps)


def write_trajectory_snapshot(snapshot_path: Path, request: TrajectoryRenderRequest) -> None:
    """Записать лёгкий снапшот (пути к JSONL + метаданные) на диск."""
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = snapshot_path.with_suffix(".tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(request, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(snapshot_path)


def load_trajectory_snapshot(snapshot_path: Path) -> TrajectoryRenderRequest:
    with snapshot_path.open("rb") as handle:
        request = pickle.load(handle)
    if not isinstance(request, TrajectoryRenderRequest):
        raise TypeError(f"Expected TrajectoryRenderRequest, got {type(request)!r}")
    return request


def start_trajectory_render_worker(
    *,
    ctx,
    output_dir: Path,
    daemon: bool = True,
) -> tuple[Queue, Any]:
    """Создать очередь и отдельный процесс для рендера траекторий."""
    task_queue: Queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_render_worker_main,
        args=(task_queue, output_dir),
        daemon=daemon,
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
            dropped = task_queue.get_nowait()
            _discard_queued_snapshot(dropped)
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


def submit_trajectory_render(task_queue: Queue | None, snapshot_path: Path) -> None:
    """Передать путь к файлу снапшота в очередь рендера."""
    if task_queue is None:
        return

    path_str = str(snapshot_path.resolve())
    try:
        task_queue.put_nowait(path_str)
    except queue.Full:
        try:
            dropped = task_queue.get_nowait()
            _discard_queued_snapshot(dropped)
        except queue.Empty:
            pass
        try:
            task_queue.put_nowait(path_str)
        except queue.Full:
            snapshot_path.unlink(missing_ok=True)


def cleanup_snapshots_dir(output_dir: Path) -> None:
    """Удалить каталог временных снапшотов после остановки воркера."""
    snapshots_root = snapshots_dir(output_dir)
    if not snapshots_root.is_dir():
        return
    for path in snapshots_root.iterdir():
        path.unlink(missing_ok=True)
    snapshots_root.rmdir()


def _discard_queued_snapshot(task: Any) -> None:
    if task == STOP_SENTINEL:
        return
    Path(str(task)).unlink(missing_ok=True)


def _render_worker_main(task_queue: Queue, output_dir: Path) -> None:
    """Фоновый воркер, который строит HTML-файлы траекторий."""
    from diplom.train.cpu_profiling import (
        start_process_cprofile_if_enabled,
        stop_process_cprofile_if_running,
    )
    from diplom.train.memory_profiling import (
        TRAJECTORY_PROCESS_NAME,
        start_process_memray_if_enabled,
        stop_process_memray_if_running,
    )

    start_process_memray_if_enabled(TRAJECTORY_PROCESS_NAME)
    start_process_cprofile_if_enabled(TRAJECTORY_PROCESS_NAME)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        while True:
            task = task_queue.get()
            if task == STOP_SENTINEL:
                break

            snapshot_path = Path(str(task))
            try:
                request = load_trajectory_snapshot(snapshot_path)
                _render_snapshot(request, output_dir)
            except Exception:  # noqa: BLE001
                traceback.print_exc()
            finally:
                snapshot_path.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    finally:
        stop_process_memray_if_running()
        stop_process_cprofile_if_running()


def _episode_from_file_ref(ref: EpisodeFileRef) -> EpisodeVizData:
    return EpisodeVizData(
        env_idx=ref.env_idx,
        steps=load_viz_steps_jsonl(ref.steps_path, step_count=ref.step_count),
        target_position=np.asarray(ref.target_position, dtype=np.float32),
        label=ref.label,
    )


def _snapshot_bounds(request: TrajectoryRenderRequest) -> TrajectoryBounds:
    if request.world_bounds is not None:
        wb = request.world_bounds
        return TrajectoryBounds(
            xmin=wb.x_min, xmax=wb.x_max,
            ymin=wb.y_min, ymax=wb.y_max,
            zmin=wb.z_min, zmax=wb.z_max,
        )

    min_xyz: np.ndarray | None = None
    max_xyz: np.ndarray | None = None

    for episodes in request.history.values():
        for ref in episodes:
            min_xyz, max_xyz = accumulate_position_extents(ref.steps_path, min_xyz, max_xyz)
            min_xyz, max_xyz = include_position_in_extents(ref.target_position, min_xyz, max_xyz)

    for steps_path in request.current_steps_paths.values():
        min_xyz, max_xyz = accumulate_position_extents(steps_path, min_xyz, max_xyz)
        last_target = load_last_target_from_jsonl(steps_path)
        if last_target is not None:
            min_xyz, max_xyz = include_position_in_extents(last_target, min_xyz, max_xyz)

    return compute_trajectory_bounds_from_extents(min_xyz, max_xyz)


def _render_snapshot(
    request: TrajectoryRenderRequest,
    output_dir: Path,
) -> None:
    """Построить и сохранить HTML для снимка траекторий."""
    bounds = _snapshot_bounds(request)

    for env_idx in range(request.n_envs):
        episode_refs = request.history.get(env_idx, [])
        current_path = request.current_steps_paths.get(env_idx)
        current_step_count = request.current_step_counts.get(env_idx, 0)
        current_env_steps = (
            load_viz_steps_jsonl(current_path, step_count=current_step_count)
            if current_path
            else []
        )
        if not episode_refs and not current_env_steps:
            continue

        history_items = [_episode_from_file_ref(episode) for episode in episode_refs]
        try:
            live_step_count = request.current_step_counts.get(env_idx, len(current_env_steps))
            fig = _build_figure_for_env(
                env_idx=env_idx,
                history=history_items,
                current_steps=current_env_steps,
                live_step_count=live_step_count,
                bounds=bounds,
                num_timesteps=request.num_timesteps,
                episode_count=request.episode_counts.get(env_idx, 0),
            )
            save_figure(fig, output_dir / f"env_{env_idx:03d}.html")
        finally:
            del history_items, current_env_steps


def _build_figure_for_env(
    *,
    env_idx: int,
    history: list[EpisodeVizData],
    current_steps: list[dict[str, Any]],
    live_step_count: int,
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
