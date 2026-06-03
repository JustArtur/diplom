from __future__ import annotations

import atexit
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from memray import Tracker

MEMRAY_DIR_ENV = "DIPLOM_MEMRAY_DIR"
MEMRAY_MAIN_PID_ENV = "DIPLOM_MEMRAY_MAIN_PID"
MEMRAY_NATIVE_ENV = "DIPLOM_MEMRAY_NATIVE"
MEMRAY_PROFILE_MAIN_ENV = "DIPLOM_MEMRAY_PROFILE_MAIN"
MEMRAY_PROFILE_ENVS_ENV = "DIPLOM_MEMRAY_PROFILE_ENVS"
MEMRAY_PROFILE_TRAJECTORY_ENV = "DIPLOM_MEMRAY_PROFILE_TRAJECTORY"

MEMRAY_SUBDIR = "memray"
MAIN_PROCESS_NAME = "main"
TRAJECTORY_PROCESS_NAME = "trajectory_render"

_active_tracker: Tracker | None = None
_tracker_started: bool = False


@dataclass(frozen=True, slots=True)
class MemrayProfileTargets:
    # Какие процессы профилировать memray (по умолчанию всё выключено)

    main: bool = False
    envs: bool = False
    trajectory: bool = False

    def any_enabled(self) -> bool:
        return self.main or self.envs or self.trajectory

    def needs_child_hooks(self) -> bool:
        return self.envs or self.trajectory


def env_process_name(env_idx: int | None) -> str:
    if env_idx is None:
        return "env_unknown"
    return f"env_{env_idx:03d}"


def memray_bin_path(memray_dir: Path, process_name: str) -> Path:
    return memray_dir / f"{process_name}.bin"


def _native_traces_enabled() -> bool:
    return os.environ.get(MEMRAY_NATIVE_ENV) == "1"


def _env_flag_enabled(env_name: str) -> bool:
    return os.environ.get(env_name) == "1"


def _role_profiling_enabled(process_name: str) -> bool:
    if process_name.startswith("env_"):
        return _env_flag_enabled(MEMRAY_PROFILE_ENVS_ENV)
    if process_name == TRAJECTORY_PROCESS_NAME:
        return _env_flag_enabled(MEMRAY_PROFILE_TRAJECTORY_ENV)
    return False


def _apply_profile_targets_env(targets: MemrayProfileTargets) -> None:
    if targets.main:
        os.environ[MEMRAY_PROFILE_MAIN_ENV] = "1"
    if targets.envs:
        os.environ[MEMRAY_PROFILE_ENVS_ENV] = "1"
    if targets.trajectory:
        os.environ[MEMRAY_PROFILE_TRAJECTORY_ENV] = "1"


def _clear_profile_targets_env() -> None:
    for key in (
        MEMRAY_PROFILE_MAIN_ENV,
        MEMRAY_PROFILE_ENVS_ENV,
        MEMRAY_PROFILE_TRAJECTORY_ENV,
    ):
        os.environ.pop(key, None)


def _load_tracker_class() -> type[Tracker]:
    try:
        from memray import Tracker
    except ImportError as exc:
        raise ImportError(
            "memray не найден. Установите dev-зависимости:\n"
            "  poetry install --with dev"
        ) from exc
    return Tracker


def stop_process_memray_if_running() -> None:
    global _active_tracker, _tracker_started

    if _active_tracker is None:
        return

    tracker = _active_tracker
    _active_tracker = None
    _tracker_started = False

    try:
        tracker.__exit__(None, None, None)
    except Exception:  # noqa: BLE001
        pass


def start_process_memray_if_enabled(process_name: str) -> None:
    global _active_tracker, _tracker_started

    if _tracker_started:
        return

    memray_dir_str = os.environ.get(MEMRAY_DIR_ENV)
    if not memray_dir_str:
        return

    if not _role_profiling_enabled(process_name):
        return

    main_pid_str = os.environ.get(MEMRAY_MAIN_PID_ENV)
    if main_pid_str is not None and os.getpid() == int(main_pid_str):
        if process_name.startswith("env_"):
            return

    memray_dir = Path(memray_dir_str)
    memray_dir.mkdir(parents=True, exist_ok=True)
    bin_path = memray_bin_path(memray_dir, process_name)

    Tracker = _load_tracker_class()
    _active_tracker = Tracker(bin_path, native_traces=_native_traces_enabled())
    _active_tracker.__enter__()
    _tracker_started = True

    atexit.register(stop_process_memray_if_running)
    print(f"memray{process_name} (pid={os.getpid()}) -> {bin_path}")


@contextmanager
def multiprocess_memray_session(
    run_dir: Path,
    *,
    targets: MemrayProfileTargets,
    native_traces: bool = False,
) -> Iterator[Path]:
    # Включить memray в дочерних процессах через переменные окружения
    memray_dir = run_dir / MEMRAY_SUBDIR
    memray_dir.mkdir(parents=True, exist_ok=True)

    os.environ[MEMRAY_DIR_ENV] = str(memray_dir.resolve())
    os.environ[MEMRAY_MAIN_PID_ENV] = str(os.getpid())
    _apply_profile_targets_env(targets)
    if native_traces:
        os.environ[MEMRAY_NATIVE_ENV] = "1"

    try:
        yield memray_dir
    finally:
        os.environ.pop(MEMRAY_DIR_ENV, None)
        os.environ.pop(MEMRAY_MAIN_PID_ENV, None)
        os.environ.pop(MEMRAY_NATIVE_ENV, None)
        _clear_profile_targets_env()
