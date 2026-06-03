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
from diplom.viz.plotly.episode_figure import (
    build_live_training_parts,
    build_wind_overlay_traces,
    collect_trajectory_traces,
    TRAJECTORY_LIVE_WIND_OVERLAY,
    latest_sim_time,
    wind_overlay_cache_key,
)
from diplom.viz.plotly.trajectory import (
    EpisodeVizData,
    TrajectoryBounds,
    compute_trajectory_bounds_from_extents,
    save_live_trajectory_update,
)
from diplom.trajectory.steps_io import (
    EpisodeFileRef,
    accumulate_position_extents,
    include_position_in_extents,
    load_last_target_from_jsonl,
    load_viz_steps_jsonl,
)


PER_ENV_RENDER_QUEUE_SIZE = 1
STOP_SENTINEL = "__STOP__"
TRAJECTORY_RENDER_SOCKET_ENV = "DIPLOM_TRAJECTORY_RENDER_SOCKET"
RENDER_TASK_SEP = "\t"
SNAPSHOTS_SUBDIR = "_snapshots"
COMBINED_TRAJECTORY_HTML = "trajectories.html"
COMBINED_SNAPSHOT_FILENAME = "snapshot_{num_timesteps:012d}.pkl"
ENV_SNAPSHOT_FILENAME = "snapshot_{num_timesteps:012d}_env_{env_idx:03d}.pkl"
ALL_ENVS_QUEUE_SUFFIX = "all"


@dataclass(frozen=True, slots=True)
class TrajectoryRenderRequest:
    num_timesteps: int
    n_envs: int
    episode_counts: dict[int, int]
    history: dict[int, list[EpisodeFileRef]]
    current_steps_paths: dict[int, Path]
    current_step_counts: dict[int, int]
    world_bounds: WorldBounds | None = None
    wind_dataset_path: Path | None = None
    show_wind_cones: bool = False
    combined_html: bool = True


class MultiQueueRenderer:
    # Round-robin рендер: у каждого queue_id своя очередь размера 1 (только актуальный снимок).

    def __init__(self) -> None:
        self._queues: dict[str, queue.Queue[str]] = {}
        self._queue_order: list[str] = []
        self._order_lock = threading.Lock()
        self._round_robin_idx = 0

    def submit(self, queue_id: str, snapshot_path: str) -> None:
        with self._order_lock:
            if queue_id not in self._queues:
                self._queues[queue_id] = queue.Queue(maxsize=PER_ENV_RENDER_QUEUE_SIZE)
                self._queue_order.append(queue_id)
            task_queue = self._queues[queue_id]

        try:
            task_queue.put_nowait(snapshot_path)
        except queue.Full:
            dropped = task_queue.get_nowait()
            _discard_queued_snapshot(dropped)
            try:
                task_queue.put_nowait(snapshot_path)
            except queue.Full:
                _discard_queued_snapshot(snapshot_path)

    def poll_once(self) -> bool:
        with self._order_lock:
            keys = self._queue_order
            if not keys:
                return False
            start_idx = self._round_robin_idx % len(keys)
            self._round_robin_idx += 1

        for offset in range(len(keys)):
            queue_id = keys[(start_idx + offset) % len(keys)]
            task_queue = self._queues[queue_id]
            try:
                snapshot_path = task_queue.get_nowait()
            except queue.Empty:
                continue
            _process_snapshot_path(Path(snapshot_path))
            return True
        return False

    def drain(self) -> None:
        while self.poll_once():
            pass


_wind_interp = None
_wind_interp_path: Path | None = None
_wind_traces_cache: dict[tuple[Path, int], list] = {}


def snapshots_dir(output_dir: Path) -> Path:
    return Path(output_dir) / SNAPSHOTS_SUBDIR


def render_queue_id(output_dir: Path, env_idx: int | None = None) -> str:
    base = f"{Path(output_dir).resolve()}"
    if env_idx is None:
        return f"{base}::{ALL_ENVS_QUEUE_SUFFIX}"
    return f"{base}::{env_idx}"


def snapshot_path_for(
    output_dir: Path,
    num_timesteps: int,
    env_idx: int | None = None,
) -> Path:
    if env_idx is None:
        return snapshots_dir(output_dir) / COMBINED_SNAPSHOT_FILENAME.format(
            num_timesteps=num_timesteps,
        )
    return snapshots_dir(output_dir) / ENV_SNAPSHOT_FILENAME.format(
        num_timesteps=num_timesteps,
        env_idx=env_idx,
    )


def write_trajectory_snapshot(snapshot_path: Path, request: TrajectoryRenderRequest) -> None:
    # Записать лёгкий снапшот (пути к JSONL + метаданные) на диск.
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
    # Создать очередь и отдельный процесс для рендера траекторий.
    task_queue: Queue = ctx.Queue()
    process = ctx.Process(
        target=_render_worker_main,
        args=(task_queue,),
        daemon=daemon,
    )
    process.start()
    return task_queue, process


def start_shared_trajectory_render_server(*, ctx) -> tuple[Path, Any]:
    # Один воркер рендера на несколько независимых train-ppo (Unix socket).
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
        _submit_render_task(str(socket_path), STOP_SENTINEL)
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
    # Остановить процесс рендера, если он был запущен.
    if isinstance(task_queue, str):
        # Общий socket-сервер (train-parallel-ppo) останавливается снаружи.
        return

    if task_queue is None or process is None:
        return

    try:
        task_queue.put_nowait(STOP_SENTINEL)
    except queue.Full:
        pass

    process.join(timeout=5.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5.0)


def submit_trajectory_render(
    task_queue: Queue | str | None,
    snapshot_path: Path,
    *,
    queue_id: str,
) -> None:
    # Передать путь к файлу снапшота в очередь рендера (per-env, размер 1).
    if task_queue is None:
        return

    task = _format_render_task(queue_id, snapshot_path)
    if isinstance(task_queue, str):
        try:
            _submit_render_task(task_queue, task)
        except OSError as exc:
            print(f"[trajectory_render] не удалось отправить снимок: {exc}", flush=True)  # noqa: T201
            snapshot_path.unlink(missing_ok=True)
        return
    try:
        task_queue.put_nowait(task)
    except queue.Full:
        snapshot_path.unlink(missing_ok=True)


def cleanup_snapshots_dir(output_dir: Path) -> None:
    # Удалить каталог временных снапшотов после остановки воркера.
    snapshots_root = snapshots_dir(output_dir)
    if not snapshots_root.is_dir():
        return
    for path in snapshots_root.iterdir():
        path.unlink(missing_ok=True)
    snapshots_root.rmdir()


def _discard_queued_snapshot(task: Any) -> None:
    if task == STOP_SENTINEL:
        return
    dropped = Path(str(task))
    print(f"[trajectory_render] снимок отброшен (очередь переполнена): {dropped.name}", flush=True)  # noqa: T201
    dropped.unlink(missing_ok=True)


def _format_render_task(queue_id: str, snapshot_path: Path) -> str:
    return f"{queue_id}{RENDER_TASK_SEP}{snapshot_path.resolve()}"


def _parse_render_task(task: str) -> tuple[str, Path] | None:
    if task == STOP_SENTINEL:
        return None
    if RENDER_TASK_SEP not in task:
        return ("__legacy__", Path(task))
    queue_id, path_str = task.split(RENDER_TASK_SEP, 1)
    return queue_id, Path(path_str)


def _run_render_loop(
    inbox_get: Any,
    *,
    inbox_get_timeout: float = 0.05,
) -> None:
    hub = MultiQueueRenderer()
    while True:
        try:
            task = inbox_get(timeout=inbox_get_timeout)
        except queue.Empty:
            task = None

        if task == STOP_SENTINEL:
            break
        if task is not None:
            parsed = _parse_render_task(str(task))
            if parsed is not None:
                queue_id, snapshot_path = parsed
                hub.submit(queue_id, os.fspath(snapshot_path))

        hub.poll_once()
    hub.drain()


def _render_worker_main(task_queue: Queue) -> None:
    # Фоновый воркер, который строит HTML-файлы траекторий.
    from diplom.dev.profiling.cpu import (
        start_process_cprofile_if_enabled,
        stop_process_cprofile_if_running,
    )
    from diplom.dev.profiling.memory import (
        TRAJECTORY_PROCESS_NAME,
        start_process_memray_if_enabled,
        stop_process_memray_if_running,
    )

    start_process_memray_if_enabled(TRAJECTORY_PROCESS_NAME)
    start_process_cprofile_if_enabled(TRAJECTORY_PROCESS_NAME)
    try:
        _run_render_loop(task_queue.get)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    finally:
        _close_wind_interp_cache()
        stop_process_memray_if_running()
        stop_process_cprofile_if_running()


def _shared_render_server_main(socket_path: Path) -> None:
    inbox: queue.Queue[str | None] = queue.Queue()
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
                inbox.put(payload)
                if payload == STOP_SENTINEL:
                    stop_event.set()
                    break
        finally:
            server.close()

    accept_thread = threading.Thread(target=accept_loop, daemon=True)
    accept_thread.start()
    try:
        _run_render_loop(inbox.get)
    finally:
        _close_wind_interp_cache()
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


def _submit_render_task(socket_path: str, task: str) -> None:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(socket_path)
        client.sendall(f"{task}\n".encode())


def _wait_socket_ready(socket_path: Path, timeout_s: float = 10.0) -> None:
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if socket_path.exists():
            return
        time.sleep(0.05)
    raise TimeoutError(f"trajectory render server не поднялся: {socket_path}")


def _episode_from_file_ref(ref: EpisodeFileRef) -> EpisodeVizData:
    steps_path = ref.steps_path.resolve()
    return EpisodeVizData(
        env_idx=ref.env_idx,
        steps=load_viz_steps_jsonl(steps_path, step_count=ref.step_count),
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
            min_xyz, max_xyz = accumulate_position_extents(
                ref.steps_path, min_xyz, max_xyz, step_count=ref.step_count,
            )
            min_xyz, max_xyz = include_position_in_extents(ref.target_position, min_xyz, max_xyz)

    for env_idx, steps_path in request.current_steps_paths.items():
        step_count = request.current_step_counts.get(env_idx, 0)
        min_xyz, max_xyz = accumulate_position_extents(
            steps_path, min_xyz, max_xyz, step_count=step_count or None,
        )
        last_target = load_last_target_from_jsonl(steps_path)
        if last_target is not None:
            min_xyz, max_xyz = include_position_in_extents(last_target, min_xyz, max_xyz)

    return compute_trajectory_bounds_from_extents(min_xyz, max_xyz)


def _resolve_wind_overlay(
    *,
    wind_dataset_path: Path | None,
    show_wind_cones: bool,
    current_steps: list[dict[str, Any]],
    history: list[EpisodeVizData],
) -> tuple[list | None, int | None]:
    if not show_wind_cones or wind_dataset_path is None:
        return None, None
    sim_time = latest_sim_time(current_steps, history)
    if sim_time is None:
        return None, None

    wind_key = wind_overlay_cache_key(sim_time)
    cache_key = (wind_dataset_path.resolve(), wind_key)
    cached = _wind_traces_cache.get(cache_key)
    if cached is not None:
        return cached, wind_key

    interpolator = _get_wind_interp(wind_dataset_path)
    traces = build_wind_overlay_traces(interpolator, sim_time, **TRAJECTORY_LIVE_WIND_OVERLAY)
    _wind_traces_cache[cache_key] = traces
    return traces, wind_key


def _render_snapshot(
    request: TrajectoryRenderRequest,
    output_dir: Path,
) -> None:
    # Построить и сохранить HTML для снимка траекторий.
    bounds = _snapshot_bounds(request)
    env_indices = sorted(
        set(request.history)
        | set(request.current_steps_paths)
    )
    if not env_indices:
        return

    if request.combined_html:
        _render_training_combined(request, output_dir, env_indices, bounds)
        return

    for env_idx in env_indices:
        _render_env_html(request, output_dir, env_idx, bounds)


def _load_env_viz_data(
    request: TrajectoryRenderRequest,
    env_idx: int,
) -> tuple[list[EpisodeVizData], list[dict[str, Any]], int]:
    episode_refs = request.history.get(env_idx, [])
    current_path = request.current_steps_paths.get(env_idx)
    current_step_count = request.current_step_counts.get(env_idx, 0)
    current_env_steps = (
        load_viz_steps_jsonl(current_path, step_count=current_step_count)
        if current_path
        else []
    )
    history_items = [_episode_from_file_ref(episode) for episode in episode_refs]
    live_step_count = request.current_step_counts.get(env_idx, len(current_env_steps))
    return history_items, current_env_steps, live_step_count


def _render_training_combined(
    request: TrajectoryRenderRequest,
    output_dir: Path,
    env_indices: list[int],
    bounds: TrajectoryBounds,
) -> None:
    all_traces: list = []
    wind_current_steps: list[dict[str, Any]] = []
    wind_history: list[EpisodeVizData] = []
    latest_wind_time: np.datetime64 | None = None

    for env_idx in env_indices:
        history_items, current_env_steps, live_step_count = _load_env_viz_data(request, env_idx)
        if not history_items and not current_env_steps:
            continue
        all_traces.extend(
            collect_trajectory_traces(
                env_idx=env_idx,
                history=history_items,
                current_steps=current_env_steps,
                live_step_count=live_step_count,
            )
        )
        sim_time = latest_sim_time(current_env_steps, history_items)
        if sim_time is not None and (
            latest_wind_time is None or sim_time > latest_wind_time
        ):
            latest_wind_time = sim_time
            wind_current_steps = current_env_steps
            wind_history = history_items

    if not all_traces:
        return

    wind_traces, wind_key = _resolve_wind_overlay(
        wind_dataset_path=request.wind_dataset_path,
        show_wind_cones=request.show_wind_cones,
        current_steps=wind_current_steps,
        history=wind_history,
    )
    total_episodes = sum(request.episode_counts.get(env_idx, 0) for env_idx in env_indices)
    title = (
        f"обучение · {request.n_envs} сред · "
        f"шаг {request.num_timesteps:,} · "
        f"завершено эпизодов: {total_episodes}"
    )
    save_live_trajectory_update(
        output_dir / COMBINED_TRAJECTORY_HTML,
        generation=request.num_timesteps,
        trajectory_traces=all_traces,
        title=title,
        bounds=bounds,
        wind_traces=wind_traces,
        wind_key=wind_key,
    )


def _render_env_html(
    request: TrajectoryRenderRequest,
    output_dir: Path,
    env_idx: int,
    bounds: TrajectoryBounds,
) -> None:
    history_items, current_env_steps, live_step_count = _load_env_viz_data(request, env_idx)
    if not history_items and not current_env_steps:
        return

    try:
        wind_traces, wind_key = _resolve_wind_overlay(
            wind_dataset_path=request.wind_dataset_path,
            show_wind_cones=request.show_wind_cones,
            current_steps=current_env_steps,
            history=history_items,
        )
        parts = build_live_training_parts(
            env_idx=env_idx,
            history=history_items,
            current_steps=current_env_steps,
            live_step_count=live_step_count,
            bounds=bounds,
            num_timesteps=request.num_timesteps,
            episode_count=request.episode_counts.get(env_idx, 0),
            wind_traces=wind_traces,
            wind_key=wind_key,
        )
        save_live_trajectory_update(
            output_dir / f"env_{env_idx:03d}.html",
            generation=request.num_timesteps,
            trajectory_traces=parts.trajectory_traces,
            title=parts.title,
            bounds=parts.bounds,
            wind_traces=parts.wind_traces,
            wind_key=parts.wind_key,
        )
    finally:
        del history_items, current_env_steps


def _close_wind_interp_cache() -> None:
    global _wind_interp, _wind_interp_path, _wind_traces_cache
    if _wind_interp is not None:
        _wind_interp.close()
    _wind_interp = None
    _wind_interp_path = None
    _wind_traces_cache.clear()


def _get_wind_interp(path: Path):
    global _wind_interp, _wind_interp_path
    resolved = path.resolve()
    if _wind_interp is not None and _wind_interp_path == resolved:
        return _wind_interp
    _close_wind_interp_cache()
    from diplom.config import WindConfig
    from diplom.wind.factory import build_wind_interpolator

    _wind_interp = build_wind_interpolator(WindConfig(path=resolved))
    _wind_interp_path = resolved
    return _wind_interp

