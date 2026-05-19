"""Профилирование обучения PPO: cProfile (CPU) и memray (память)."""

from __future__ import annotations

import cProfile
import pstats
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from diplom.config import AppConfig
from diplom.train.cpu_profiling import (
    CPROFILE_SUBDIR,
    cprofile_prof_path,
    multiprocess_cprofile_session,
)
from diplom.train.memory_profiling import (
    MAIN_PROCESS_NAME,
    MEMRAY_SUBDIR,
    MemrayProfileTargets,
    memray_bin_path,
    multiprocess_memray_session,
)

if TYPE_CHECKING:
    from memray import Tracker

# Одна среда в DummyVecEnv — для profile-ppo-cpu и режима --single-process.
PROFILE_N_ENVS = 1
PROFILE_PROF_FILENAME = "profile.prof"
PROFILE_MEMRAY_BIN = "memray.bin"
PROFILE_MEMRAY_HTML = "memray.html"


@dataclass(frozen=True, slots=True)
class MemrayProcessReport:
    process_name: str
    bin_path: Path
    html_path: Path | None = None


@dataclass(frozen=True, slots=True)
class CprofileProcessReport:
    process_name: str
    prof_path: Path


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


def _finalize_cprofile_prof(
    prof_path: Path,
    *,
    top_lines: int,
    sort_by: str,
    print_stats: bool,
) -> CprofileProcessReport:
    process_name = prof_path.stem
    if print_stats:
        print(f"\n[profile-ppo-cpu] === {process_name} ===")  # noqa: T201
        stats = pstats.Stats(str(prof_path))
        stats.strip_dirs().sort_stats(sort_by)
        stats.print_stats(top_lines)

    return CprofileProcessReport(process_name=process_name, prof_path=prof_path)


def run_cprofile_train(
    config: AppConfig,
    *,
    output: Path | None = None,
    top_lines: int = 40,
    sort_by: str = "cumulative",
    multiprocess: bool = True,
    profile_targets: MemrayProfileTargets | None = None,
    print_stats: bool = True,
) -> tuple[Path, list[CprofileProcessReport]]:
    """cProfile обучения: один процесс (DummyVecEnv) или выбранные процессы run-а."""
    from diplom.train.ppo_runner import train_ppo
    from diplom.train.run_dirs import next_run_dir

    targets = profile_targets or MemrayProfileTargets()
    if not targets.any_enabled():
        raise ValueError(
            "Укажите хотя бы один флаг: --profile-main, --profile-envs или --profile-trajectory"
        )

    run_dir = next_run_dir(config.training.logdir)
    n_envs = max(1, config.training.n_envs)

    if multiprocess:
        return _run_cprofile_train_multiprocess(
            config,
            run_dir=run_dir,
            n_envs=n_envs,
            top_lines=top_lines,
            sort_by=sort_by,
            print_stats=print_stats,
            profile_targets=targets,
        )

    if targets.envs and not targets.main:
        raise ValueError(
            "Профиль env_* доступен только с SubprocVecEnv (без --single-process). "
            "Используйте --profile-main или запустите без --single-process."
        )

    prof_path = output.resolve() if output is not None else run_dir / PROFILE_PROF_FILENAME
    prof_path.parent.mkdir(parents=True, exist_ok=True)

    print(  # noqa: T201
        f"[profile-ppo-cpu] Одна среда (n_envs={PROFILE_N_ENVS}), DummyVecEnv, "
        f"профили: {_describe_profile_targets(targets, n_envs=n_envs, multiprocess=False)}"
    )

    profile_main_in_process = targets.main or targets.envs

    def _run_training() -> None:
        train_ppo(config, force_dummy_vec_env=True, run_dir=run_dir)

    if targets.needs_child_hooks():
        with multiprocess_cprofile_session(run_dir, targets=targets):
            if profile_main_in_process:
                print(f"[profile-ppo-cpu] Главный процесс → {prof_path}")  # noqa: T201
                profiler = cProfile.Profile()
                profiler.enable()
                try:
                    _run_training()
                finally:
                    profiler.disable()
                    profiler.dump_stats(str(prof_path))
            else:
                _run_training()
    elif profile_main_in_process:
        print(f"[profile-ppo-cpu] cProfile → {prof_path}")  # noqa: T201
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            _run_training()
        finally:
            profiler.disable()
            profiler.dump_stats(str(prof_path))
    else:
        _run_training()

    reports: list[CprofileProcessReport] = []
    if profile_main_in_process:
        print(f"[profile-ppo-cpu] Профиль сохранён: {prof_path}")  # noqa: T201
        reports.append(
            _finalize_cprofile_prof(
                prof_path, top_lines=top_lines, sort_by=sort_by, print_stats=print_stats
            )
        )

    if targets.trajectory and targets.needs_child_hooks():
        cprofile_dir = run_dir / CPROFILE_SUBDIR
        child_profs = sorted(
            path for path in cprofile_dir.glob("*.prof") if path.stem != MAIN_PROCESS_NAME
        )
        reports.extend(
            _finalize_cprofile_prof(path, top_lines=top_lines, sort_by=sort_by, print_stats=print_stats)
            for path in child_profs
        )

    if reports and print_stats:
        print(  # noqa: T201
            f"[profile-ppo-cpu] Подробнее: python -m pstats {reports[0].prof_path}"
        )

    return run_dir, reports


def _run_cprofile_train_multiprocess(
    config: AppConfig,
    *,
    run_dir: Path,
    n_envs: int,
    top_lines: int,
    sort_by: str,
    print_stats: bool,
    profile_targets: MemrayProfileTargets,
) -> tuple[Path, list[CprofileProcessReport]]:
    from diplom.train.ppo_runner import train_ppo

    if config.training.use_worker_policy_rollout:
        vec_label = "PolicyShmemSubprocVecEnv"
    else:
        vec_label = "ShmemSubprocVecEnv"
    print(  # noqa: T201
        f"[profile-ppo-cpu] {vec_label} n_envs={n_envs}, "
        f"профили: {_describe_profile_targets(profile_targets, n_envs=n_envs, multiprocess=True)}"
    )

    def _run_training() -> None:
        train_ppo(config, force_dummy_vec_env=False, run_dir=run_dir)

    with multiprocess_cprofile_session(run_dir, targets=profile_targets) as cprofile_dir:
        if profile_targets.main:
            main_prof = cprofile_prof_path(cprofile_dir, MAIN_PROCESS_NAME)
            print(f"[profile-ppo-cpu] Главный процесс → {main_prof}")  # noqa: T201
            profiler = cProfile.Profile()
            profiler.enable()
            try:
                _run_training()
            finally:
                profiler.disable()
                profiler.dump_stats(str(main_prof))
        else:
            _run_training()

    prof_files = sorted(cprofile_dir.glob("*.prof"))
    if not prof_files:
        print("[profile-ppo-cpu] Нет .prof файлов cProfile", file=sys.stderr)  # noqa: T201
        return run_dir, []

    print(f"[profile-ppo-cpu] Профили CPU ({len(prof_files)}) в {cprofile_dir}:")  # noqa: T201
    for path in prof_files:
        print(f"  - {path.name}")  # noqa: T201

    reports = [
        _finalize_cprofile_prof(path, top_lines=top_lines, sort_by=sort_by, print_stats=print_stats)
        for path in prof_files
    ]
    if reports and print_stats:
        print(  # noqa: T201
            f"[profile-ppo-cpu] Подробнее: python -m pstats {reports[0].prof_path}"
        )
    return run_dir, reports


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


def _finalize_memray_bin(
    bin_path: Path,
    *,
    skip_flamegraph: bool,
    print_table: bool,
    flamegraph: Path | None = None,
) -> MemrayProcessReport:
    process_name = bin_path.stem
    if print_table:
        print(f"\n[profile-ppo-mem] === {process_name} ===")  # noqa: T201
        print_memray_table(bin_path)

    html_path: Path | None = None
    if not skip_flamegraph:
        html_target = flamegraph.resolve() if flamegraph is not None else bin_path.with_suffix(".html")
        if write_memray_flamegraph(bin_path, html_target):
            html_path = html_target
            print(f"[profile-ppo-mem] Flame graph ({process_name}): {html_path}")  # noqa: T201
        else:
            print(  # noqa: T201
                f"[profile-ppo-mem] Не удалось построить flame graph для {bin_path}",
                file=sys.stderr,
            )
    return MemrayProcessReport(process_name=process_name, bin_path=bin_path, html_path=html_path)


def _describe_profile_targets(targets: MemrayProfileTargets, *, n_envs: int, multiprocess: bool) -> str:
    parts: list[str] = []
    if targets.main:
        parts.append("main")
    if targets.envs:
        if multiprocess:
            parts.append(f"{n_envs}×env_*")
        else:
            parts.append("main (среда в том же процессе)")
    if targets.trajectory:
        parts.append("trajectory_render")
    return ", ".join(parts) if parts else "ничего"


def run_memray_train(
    config: AppConfig,
    *,
    output: Path | None = None,
    flamegraph: Path | None = None,
    skip_flamegraph: bool = False,
    native_traces: bool = False,
    print_table: bool = True,
    multiprocess: bool = True,
    profile_targets: MemrayProfileTargets | None = None,
) -> tuple[Path, list[MemrayProcessReport]]:
    """memray-отчёт обучения: один процесс (DummyVecEnv) или все процессы run-а."""
    from diplom.train.ppo_runner import train_ppo
    from diplom.train.run_dirs import next_run_dir

    targets = profile_targets or MemrayProfileTargets()
    if not targets.any_enabled():
        raise ValueError(
            "Укажите хотя бы один флаг: --profile-main, --profile-envs или --profile-trajectory"
        )

    Tracker = _load_tracker()
    run_dir = next_run_dir(config.training.logdir)
    n_envs = max(1, config.training.n_envs)

    if multiprocess:
        return _run_memray_train_multiprocess(
            config,
            run_dir=run_dir,
            Tracker=Tracker,
            n_envs=n_envs,
            skip_flamegraph=skip_flamegraph,
            native_traces=native_traces,
            print_table=print_table,
            profile_targets=targets,
        )

    if targets.envs and not targets.main:
        raise ValueError(
            "Профиль env_* доступен только с SubprocVecEnv (без --single-process). "
            "Используйте --profile-main или запустите без --single-process."
        )

    bin_path = output.resolve() if output is not None else run_dir / PROFILE_MEMRAY_BIN
    bin_path.parent.mkdir(parents=True, exist_ok=True)

    print(  # noqa: T201
        f"[profile-ppo-mem] Одна среда (n_envs={PROFILE_N_ENVS}), DummyVecEnv, "
        f"профили: {_describe_profile_targets(targets, n_envs=n_envs, multiprocess=False)}"
    )
    if native_traces:
        print("[profile-ppo-mem] native_traces=True (PyTorch, NumPy C-расширения)")  # noqa: T201

    profile_main_in_process = targets.main or targets.envs

    def _run_training() -> None:
        train_ppo(config, force_dummy_vec_env=True, run_dir=run_dir)

    if targets.needs_child_hooks():
        with multiprocess_memray_session(run_dir, targets=targets, native_traces=native_traces):
            if profile_main_in_process:
                print(f"[profile-ppo-mem] Главный процесс → {bin_path}")  # noqa: T201
                with Tracker(bin_path, native_traces=native_traces):
                    _run_training()
            else:
                _run_training()
    elif profile_main_in_process:
        print(f"[profile-ppo-mem] memray → {bin_path}")  # noqa: T201
        with Tracker(bin_path, native_traces=native_traces):
            _run_training()
    else:
        _run_training()

    if profile_main_in_process:
        print(f"[profile-ppo-mem] Снимок памяти: {bin_path}")  # noqa: T201
        html_override = flamegraph.resolve() if flamegraph is not None else None
        if html_override is None and not skip_flamegraph:
            html_override = bin_path.with_name(PROFILE_MEMRAY_HTML)
        reports = [
            _finalize_memray_bin(
                bin_path,
                skip_flamegraph=skip_flamegraph,
                print_table=print_table,
                flamegraph=html_override,
            )
        ]
    else:
        reports = []

    if targets.trajectory and targets.needs_child_hooks():
        memray_dir = run_dir / MEMRAY_SUBDIR
        child_bins = sorted(
            path for path in memray_dir.glob("*.bin") if path.stem != MAIN_PROCESS_NAME
        )
        reports.extend(
            _finalize_memray_bin(path, skip_flamegraph=skip_flamegraph, print_table=print_table)
            for path in child_bins
        )

    return run_dir, reports


def _run_memray_train_multiprocess(
    config: AppConfig,
    *,
    run_dir: Path,
    Tracker: type[Tracker],
    n_envs: int,
    skip_flamegraph: bool,
    native_traces: bool,
    print_table: bool,
    profile_targets: MemrayProfileTargets,
) -> tuple[Path, list[MemrayProcessReport]]:
    from diplom.train.ppo_runner import train_ppo

    print(  # noqa: T201
        f"[profile-ppo-mem] SubprocVecEnv n_envs={n_envs}, "
        f"профили: {_describe_profile_targets(profile_targets, n_envs=n_envs, multiprocess=True)}"
    )
    if native_traces:
        print("[profile-ppo-mem] native_traces=True (PyTorch, NumPy C-расширения)")  # noqa: T201

    def _run_training() -> None:
        train_ppo(config, force_dummy_vec_env=False, run_dir=run_dir)

    with multiprocess_memray_session(
        run_dir, targets=profile_targets, native_traces=native_traces
    ) as memray_dir:
        if profile_targets.main:
            main_bin = memray_bin_path(memray_dir, MAIN_PROCESS_NAME)
            print(f"[profile-ppo-mem] Главный процесс → {main_bin}")  # noqa: T201
            with Tracker(main_bin, native_traces=native_traces):
                _run_training()
        else:
            _run_training()

    bin_files = sorted(memray_dir.glob("*.bin"))
    if not bin_files:
        print("[profile-ppo-mem] Нет .bin файлов memray", file=sys.stderr)  # noqa: T201
        return run_dir, []

    print(f"[profile-ppo-mem] Снимки памяти ({len(bin_files)}) в {memray_dir}:")  # noqa: T201
    for path in bin_files:
        print(f"  - {path.name}")  # noqa: T201

    reports = [
        _finalize_memray_bin(path, skip_flamegraph=skip_flamegraph, print_table=print_table)
        for path in bin_files
    ]
    return run_dir, reports
