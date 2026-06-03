# Профилирование CPU (cProfile) в главном и дочерних процессах обучения.

from __future__ import annotations

import atexit
import cProfile
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from diplom.dev.profiling.memory import (
    TRAJECTORY_PROCESS_NAME,
    MemrayProfileTargets,
)

CPROFILE_DIR_ENV = "DIPLOM_CPROFILE_DIR"
CPROFILE_MAIN_PID_ENV = "DIPLOM_CPROFILE_MAIN_PID"
CPROFILE_PROFILE_MAIN_ENV = "DIPLOM_CPROFILE_PROFILE_MAIN"
CPROFILE_PROFILE_ENVS_ENV = "DIPLOM_CPROFILE_PROFILE_ENVS"
CPROFILE_PROFILE_TRAJECTORY_ENV = "DIPLOM_CPROFILE_PROFILE_TRAJECTORY"

CPROFILE_SUBDIR = "cprofile"

_active_profiler: cProfile.Profile | None = None
_profiler_started: bool = False


def cprofile_prof_path(cprofile_dir: Path, process_name: str) -> Path:
    return cprofile_dir / f"{process_name}.prof"


def _env_flag_enabled(env_name: str) -> bool:
    return os.environ.get(env_name) == "1"


def _role_profiling_enabled(process_name: str) -> bool:
    if process_name.startswith("env_"):
        return _env_flag_enabled(CPROFILE_PROFILE_ENVS_ENV)
    if process_name == TRAJECTORY_PROCESS_NAME:
        return _env_flag_enabled(CPROFILE_PROFILE_TRAJECTORY_ENV)
    return False


def _apply_profile_targets_env(targets: MemrayProfileTargets) -> None:
    if targets.main:
        os.environ[CPROFILE_PROFILE_MAIN_ENV] = "1"
    if targets.envs:
        os.environ[CPROFILE_PROFILE_ENVS_ENV] = "1"
    if targets.trajectory:
        os.environ[CPROFILE_PROFILE_TRAJECTORY_ENV] = "1"


def _clear_profile_targets_env() -> None:
    for key in (
        CPROFILE_PROFILE_MAIN_ENV,
        CPROFILE_PROFILE_ENVS_ENV,
        CPROFILE_PROFILE_TRAJECTORY_ENV,
    ):
        os.environ.pop(key, None)


def stop_process_cprofile_if_running() -> None:
    # Сохранить cProfile и выключить трекер в текущем процессе.
    global _active_profiler, _profiler_started

    if _active_profiler is None:
        return

    profiler = _active_profiler
    _active_profiler = None
    _profiler_started = False

    profiler.disable()
    prof_path_str = getattr(profiler, "_diplom_prof_path", None)
    if prof_path_str:
        Path(prof_path_str).parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(prof_path_str)
        print(f"[cprofile] сохранён {prof_path_str}")  # noqa: T201


def start_process_cprofile_if_enabled(process_name: str) -> None:
    # Запустить cProfile в текущем процессе (один раз на PID).
    global _active_profiler, _profiler_started

    if _profiler_started:
        return

    cprofile_dir_str = os.environ.get(CPROFILE_DIR_ENV)
    if not cprofile_dir_str:
        return

    if not _role_profiling_enabled(process_name):
        return

    main_pid_str = os.environ.get(CPROFILE_MAIN_PID_ENV)
    if main_pid_str is not None and os.getpid() == int(main_pid_str):
        if process_name.startswith("env_"):
            return

    cprofile_dir = Path(cprofile_dir_str)
    cprofile_dir.mkdir(parents=True, exist_ok=True)
    prof_path = cprofile_prof_path(cprofile_dir, process_name)

    profiler = cProfile.Profile()
    profiler._diplom_prof_path = str(prof_path)  # type: ignore[attr-defined]
    profiler.enable()
    _active_profiler = profiler
    _profiler_started = True

    atexit.register(stop_process_cprofile_if_running)
    print(f"[cprofile] {process_name} (pid={os.getpid()}) -> {prof_path}")  # noqa: T201


@contextmanager
def multiprocess_cprofile_session(
    run_dir: Path,
    *,
    targets: MemrayProfileTargets,
) -> Iterator[Path]:
    # Включить cProfile в дочерних процессах через переменные окружения.
    cprofile_dir = run_dir / CPROFILE_SUBDIR
    cprofile_dir.mkdir(parents=True, exist_ok=True)

    os.environ[CPROFILE_DIR_ENV] = str(cprofile_dir.resolve())
    os.environ[CPROFILE_MAIN_PID_ENV] = str(os.getpid())
    _apply_profile_targets_env(targets)

    try:
        yield cprofile_dir
    finally:
        os.environ.pop(CPROFILE_DIR_ENV, None)
        os.environ.pop(CPROFILE_MAIN_PID_ENV, None)
        _clear_profile_targets_env()
