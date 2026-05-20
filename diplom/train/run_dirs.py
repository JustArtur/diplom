"""Каталоги одного запуска обучения: runs/<logdir>/PPO_N/{tb,trajectories}."""

from __future__ import annotations

import re
from pathlib import Path

_RUN_SUFFIX_RE = re.compile(r"(?:^|#)PPO_(\d+)$")


def _ppo_index_from_dir_name(name: str) -> int | None:
    match = _RUN_SUFFIX_RE.search(name)
    if match is None:
        return None
    return int(match.group(1))


def _next_ppo_index(runs_root: Path) -> int:
    nums: list[int] = []
    for path in runs_root.iterdir():
        if not path.is_dir():
            continue
        idx = _ppo_index_from_dir_name(path.name)
        if idx is not None:
            nums.append(idx)
    return max(nums) + 1 if nums else 0


def _validate_run_name(run_name: str) -> None:
    if not run_name.strip():
        raise ValueError("Имя run-а не может быть пустым")
    if "/" in run_name or "\\" in run_name or "#" in run_name:
        raise ValueError("Имя run-а не должно содержать '/', '\\' или '#'")


def next_run_dir(runs_root: Path, *, run_name: str | None = None) -> Path:
    """Следующая директория run-а внутри runs_root.

    Без run_name: ``PPO_N`` (PPO_0, PPO_1, …).
    С run_name: ``{run_name}#PPO_N``.
    """
    runs_root.mkdir(parents=True, exist_ok=True)
    if run_name is not None:
        _validate_run_name(run_name)
    next_n = _next_ppo_index(runs_root)
    if run_name is None:
        return runs_root / f"PPO_{next_n}"
    return runs_root / f"{run_name}#PPO_{next_n}"
