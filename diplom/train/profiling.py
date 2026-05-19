"""Профилирование обучения PPO: cProfile (CPU) и memray (память)."""

from __future__ import annotations

import cProfile
import pstats
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from diplom.config import AppConfig

if TYPE_CHECKING:
    from memray import Tracker

# Профилирование всегда в одном процессе Python (DummyVecEnv), без SubprocVecEnv.
PROFILE_N_ENVS = 1
PROFILE_PROF_FILENAME = "profile.prof"
PROFILE_MEMRAY_BIN = "memray.bin"
PROFILE_MEMRAY_HTML = "memray.html"


class MemrayNotFoundError(ImportError):
    """memray не установлен."""


def _load_tracker() -> type[Tracker]:
    try:
        from memray import Tracker
    except ImportError as exc:
        raise MemrayNotFoundError(
            "memray не найден. Установите dev-зависимости:\n"
            "  poetry install --with dev"
        ) from exc
    return Tracker


def run_cprofile_train(
    config: AppConfig,
    *,
    output: Path | None = None,
    top_lines: int = 40,
    sort_by: str = "cumulative",
) -> Path:
    """cProfile обучения в одном процессе (DummyVecEnv, без root на macOS)."""
    from diplom.train.ppo_runner import train_ppo
    from diplom.train.run_dirs import next_run_dir

    run_dir = next_run_dir(config.training.logdir)
    prof_path = (
        output.resolve()
        if output is not None
        else run_dir / PROFILE_PROF_FILENAME
    )

    print(  # noqa: T201
        f"[profile-ppo-cpu] Одна среда (n_envs={PROFILE_N_ENVS}), DummyVecEnv, cProfile → {prof_path}"
    )
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        train_ppo(config, force_dummy_vec_env=True, run_dir=run_dir)
    finally:
        profiler.disable()
    prof_path.parent.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(str(prof_path))

    print(f"[profile-ppo-cpu] Профиль сохранён: {prof_path}")  # noqa: T201
    stats = pstats.Stats(str(prof_path))
    stats.strip_dirs().sort_stats(sort_by)
    stats.print_stats(top_lines)
    print(  # noqa: T201
        f"[profile-ppo-cpu] Подробнее: python -m pstats {prof_path}"
    )
    return prof_path


def _run_memray_cli(*args: str) -> int:
    completed = subprocess.run(
        [sys.executable, "-m", "memray", *args],
        check=False,
    )
    return completed.returncode


def print_memray_table(bin_path: Path) -> None:
    """Топ аллокаций в терминал (`memray table`)."""
    code = _run_memray_cli("table", str(bin_path.resolve()))
    if code != 0:
        print(f"[profile-ppo-mem] memray table завершился с кодом {code}", file=sys.stderr)  # noqa: T201


def write_memray_flamegraph(bin_path: Path, html_path: Path) -> bool:
    """HTML flame graph (`memray flamegraph`)."""
    html_path = html_path.resolve()
    html_path.parent.mkdir(parents=True, exist_ok=True)
    code = _run_memray_cli("flamegraph", str(bin_path.resolve()), "-o", str(html_path))
    return code == 0


def run_memray_train(
    config: AppConfig,
    *,
    output: Path | None = None,
    flamegraph: Path | None = None,
    skip_flamegraph: bool = False,
    native_traces: bool = False,
    print_table: bool = True,
) -> tuple[Path, Path | None]:
    """memray-отчёт обучения в одном процессе (DummyVecEnv)."""
    from diplom.train.ppo_runner import train_ppo
    from diplom.train.run_dirs import next_run_dir

    Tracker = _load_tracker()
    run_dir = next_run_dir(config.training.logdir)
    bin_path = output.resolve() if output is not None else run_dir / PROFILE_MEMRAY_BIN
    bin_path.parent.mkdir(parents=True, exist_ok=True)

    print(  # noqa: T201
        f"[profile-ppo-mem] Одна среда (n_envs={PROFILE_N_ENVS}), DummyVecEnv, memray → {bin_path}"
    )
    if native_traces:
        print("[profile-ppo-mem] native_traces=True (PyTorch, NumPy C-расширения)")  # noqa: T201

    with Tracker(bin_path, native_traces=native_traces):
        train_ppo(config, force_dummy_vec_env=True, run_dir=run_dir)

    print(f"[profile-ppo-mem] Снимок памяти: {bin_path}")  # noqa: T201

    if print_table:
        print_memray_table(bin_path)

    html_path: Path | None = None
    if not skip_flamegraph:
        html_target = (
            flamegraph.resolve()
            if flamegraph is not None
            else bin_path.with_name(PROFILE_MEMRAY_HTML)
        )
        if write_memray_flamegraph(bin_path, html_target):
            html_path = html_target
            print(f"[profile-ppo-mem] Flame graph: {html_path}")  # noqa: T201
        else:
            print(  # noqa: T201
                f"[profile-ppo-mem] Не удалось построить flame graph для {bin_path}",
                file=sys.stderr,
            )

    return bin_path, html_path
