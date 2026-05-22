"""Параллельный запуск нескольких train-ppo с общим воркером рендера траекторий."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from multiprocessing import get_context
from pathlib import Path

from diplom.train.trajectory_render_worker import (
    TRAJECTORY_RENDER_SOCKET_ENV,
    start_shared_trajectory_render_server,
    stop_shared_trajectory_render_server,
)

RUNNER_TOKEN = "runner"


def parse_train_parallel_argv(argv: list[str]) -> tuple[list[str], list[list[str]]]:
    """Разбить argv на глобальные опции и списки аргументов для каждого run."""
    global_args: list[str] = []
    runs: list[list[str]] = []
    current: list[str] | None = None

    for arg in argv:
        if arg == RUNNER_TOKEN:
            if current is not None:
                if not current:
                    raise ValueError("пустой блок runner: укажите аргументы train-ppo")
                runs.append(current)
            current = []
            continue
        if current is None:
            global_args.append(arg)
        else:
            current.append(arg)

    if current is not None:
        if not current:
            raise ValueError("пустой блок runner: укажите аргументы train-ppo")
        runs.append(current)

    if not runs:
        raise ValueError(
            f"нужен хотя бы один блок «{RUNNER_TOKEN}» с аргументами train-ppo, "
            f"например: diplom train-parallel-ppo {RUNNER_TOKEN} --dataset era5_..."
        )
    return global_args, runs


def _parse_jobs(global_args: list[str]) -> tuple[int, list[str]]:
    jobs = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    rest: list[str] = []
    i = 0
    while i < len(global_args):
        if global_args[i] in ("--jobs", "-j"):
            if i + 1 >= len(global_args):
                raise ValueError("--jobs требует число")
            jobs = max(1, int(global_args[i + 1]))
            i += 2
            continue
        rest.append(global_args[i])
        i += 1
    if rest:
        raise ValueError(f"неизвестные глобальные опции: {' '.join(rest)} (допустимо только --jobs)")
    return jobs, rest


def run_train_parallel_ppo(argv: list[str]) -> int:
    """Запустить несколько train-ppo; один subprocess рендера траекторий на все run."""
    from diplom.data.era5_manifest import expand_training_manifest_argv

    argv = expand_training_manifest_argv(argv)
    global_args, runs = parse_train_parallel_argv(argv)
    jobs, _ = _parse_jobs(global_args)

    ctx = get_context("spawn")
    socket_path, render_process = start_shared_trajectory_render_server(ctx=ctx)

    env = os.environ.copy()
    env[TRAJECTORY_RENDER_SOCKET_ENV] = str(socket_path)

    cli = [sys.executable, "-m", "diplom.cli", "train-ppo"]
    procs: list[subprocess.Popen[bytes]] = []
    exit_code = 0

    try:
        pending = list(enumerate(runs))
        active: dict[int, subprocess.Popen[bytes]] = {}

        while pending or active:
            while pending and len(active) < jobs:
                run_idx, run_argv = pending.pop(0)
                proc = subprocess.Popen(
                    [*cli, *run_argv],
                    env=env,
                )
                active[run_idx] = proc

            if not active:
                break

            done: list[int] = []
            for run_idx, proc in active.items():
                code = proc.poll()
                if code is None:
                    continue
                done.append(run_idx)
                if code != 0:
                    exit_code = code
                    typer_echo = f"[train-parallel-ppo] run #{run_idx + 1} завершился с кодом {code}"
                    print(typer_echo, file=sys.stderr)  # noqa: T201

            for run_idx in done:
                del active[run_idx]

            if active:
                time.sleep(0.2)

        if exit_code != 0:
            for proc in active.values():
                proc.terminate()
    finally:
        stop_shared_trajectory_render_server(socket_path, render_process)

    return exit_code
