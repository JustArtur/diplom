"""Каталоги одного запуска обучения: runs/<logdir>/PPO_N/{tb,trajectories}."""

from __future__ import annotations

from pathlib import Path


def next_run_dir(runs_root: Path) -> Path:
    """Следующая директория PPO_N внутри runs_root (PPO_0, PPO_1, …)."""
    runs_root.mkdir(parents=True, exist_ok=True)
    nums: list[int] = []
    for path in runs_root.glob("PPO_*"):
        if not path.is_dir():
            continue
        suffix = path.name.removeprefix("PPO_")
        if suffix.isdigit():
            nums.append(int(suffix))
    next_n = max(nums) + 1 if nums else 0
    return runs_root / f"PPO_{next_n}"
