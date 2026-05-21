from __future__ import annotations

import os
import pickle
import queue
import socket
import threading
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
    save_live_figure,
)
from diplom.train.trajectory_steps_io import (
    EpisodeFileRef,
    accumulate_position_extents,
    include_position_in_extents,
    load_last_target_from_jsonl,
    load_viz_steps_jsonl,
)


STOP_SENTINEL = "__STOP__"
TRAJECTORY_RENDER_SOCKET_ENV = "DIPLOM_TRAJECTORY_RENDER_SOCKET"
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
        args=(task_queue,),
        daemon=daemon,
    )
    process.start()
    return task_queue, process


def start_shared_trajectory_render_server(*, ctx) -> tuple[Path, Any]:
    """Один воркер рендера на несколько независимых train-ppo (Unix socket)."""
    socket_path = Path(os.environ.get("TMPDIR", "/tmp")) / f"diplom-traj-render-{os.getpid()}.sock"
    socket_path.unlink(missing_ok=True)
    process = ctx.Process(
        target=_shared_render_server_main,
        args=(socket_path,),
        daemon=True,
    )
    process.start()
    _wait_socket_ready(socket_path)
    return socket_path, process


def stop_shared_trajectory_render_server(socket_path: Path, process: Any | None) -> None:
    try:
        _submit_snapshot_path(str(socket_path), STOP_SENTINEL)
    except OSError:
        pass
    socket_path.unlink(missing_ok=True)
    if process is None:
        return
    process.join(timeout=5.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5.0)


def stop_trajectory_render_worker(task_queue: Queue | str | None, process: Any | None) -> None:
    """Остановить процесс рендера, если он был запущен."""
    if isinstance(task_queue, str) or task_queue is None or process is None:
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


def submit_trajectory_render(task_queue: Queue | str | None, snapshot_path: Path) -> None:
    """Передать путь к файлу снапшота в очередь рендера."""
    if task_queue is None:
        return

    path_str = str(snapshot_path.resolve())
    if isinstance(task_queue, str):
        try:
            _submit_snapshot_path(task_queue, path_str)
        except OSError:
            snapshot_path.unlink(missing_ok=True)
        return
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


def _render_worker_main(task_queue: Queue) -> None:
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
        while True:
            task = task_queue.get()
            if task == STOP_SENTINEL:
                break
            _process_snapshot_path(Path(str(task)))
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    finally:
        stop_process_memray_if_running()
        stop_process_cprofile_if_running()


def _shared_render_server_main(socket_path: Path) -> None:
    task_queue: queue.Queue[str] = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    def accept_loop() -> None:
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(os.fspath(socket_path))
            server.listen(16)
            while not stop_event.is_set():
                server.settimeout(0.5)
                try:
                    conn, _ = server.accept()
                except TimeoutError:
                    continue
                with conn:
                    payload = conn.recv(65536).decode().strip()
                if not payload:
                    continue
                if payload == STOP_SENTINEL:
                    try:
                        task_queue.put_nowait(STOP_SENTINEL)
                    except queue.Full:
                        dropped = task_queue.get_nowait()
                        _discard_queued_snapshot(dropped)
                        task_queue.put_nowait(STOP_SENTINEL)
                    stop_event.set()
                    break
                try:
                    task_queue.put_nowait(payload)
                except queue.Full:
                    dropped = task_queue.get_nowait()
                    _discard_queued_snapshot(dropped)
                    try:
                        task_queue.put_nowait(payload)
                    except queue.Full:
                        _discard_queued_snapshot(payload)
        finally:
            server.close()

    accept_thread = threading.Thread(target=accept_loop, daemon=True)
    accept_thread.start()
    try:
        while True:
            task = task_queue.get()
            if task == STOP_SENTINEL:
                break
            _process_snapshot_path(Path(task))
    finally:
        stop_event.set()
        accept_thread.join(timeout=2.0)


def _process_snapshot_path(snapshot_path: Path) -> None:
    output_dir = snapshot_path.parent.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        request = load_trajectory_snapshot(snapshot_path)
        _render_snapshot(request, output_dir)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    finally:
        snapshot_path.unlink(missing_ok=True)


def _submit_snapshot_path(socket_path: str, path_str: str) -> None:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(socket_path)
        client.sendall(f"{path_str}\n".encode())


def _wait_socket_ready(socket_path: Path, timeout_s: float = 10.0) -> None:
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if socket_path.exists():
            return
        time.sleep(0.05)
    raise TimeoutError(f"trajectory render server не поднялся: {socket_path}")


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
            save_live_figure(
                fig,
                output_dir / f"env_{env_idx:03d}.html",
                generation=request.num_timesteps,
            )
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
