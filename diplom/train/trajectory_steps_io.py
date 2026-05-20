from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Максимум точек на одну траекторию в HTML (Plotly тормозит на сотнях тысяч).
MAX_VIZ_PLOT_POINTS = 8_000


STEPS_SUBDIR = "_steps"
CURRENT_STEPS_FILENAME = "env_{env_idx:03d}_current.jsonl"
EPISODE_STEPS_FILENAME = "env_{env_idx:03d}_ep_{episode_num:06d}.jsonl"


def steps_dir(output_dir: Path) -> Path:
    return Path(output_dir) / STEPS_SUBDIR


def current_steps_path(output_dir: Path, env_idx: int) -> Path:
    return steps_dir(output_dir) / CURRENT_STEPS_FILENAME.format(env_idx=env_idx)


def episode_steps_path(output_dir: Path, env_idx: int, episode_num: int) -> Path:
    return steps_dir(output_dir) / EPISODE_STEPS_FILENAME.format(
        env_idx=env_idx,
        episode_num=episode_num,
    )


@dataclass(frozen=True, slots=True)
class EpisodeFileRef:
    steps_path: Path
    env_idx: int
    target_position: tuple[float, float, float]
    label: str
    step_count: int


class EnvStepsWriter:
    """Пишет шаги эпизода в JSONL на диск, не держа их в RAM."""

    def __init__(self, output_dir: Path, env_idx: int) -> None:
        self._output_dir = Path(output_dir)
        self.env_idx = env_idx
        self.current_path = current_steps_path(self._output_dir, env_idx)
        self.step_count = 0
        self._handle = None

    def open_current(self) -> None:
        self.current_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.current_path.open("a", encoding="utf-8")

    def append_step(self, step: dict[str, Any]) -> None:
        if self._handle is None:
            self.open_current()
        self._handle.write(json.dumps(step, ensure_ascii=False, separators=(",", ":")))
        self._handle.write("\n")
        self.step_count += 1

    def finalize_episode(self, episode_num: int) -> Path:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

        final_path = episode_steps_path(self._output_dir, self.env_idx, episode_num)
        if self.current_path.exists():
            self.current_path.replace(final_path)
        else:
            final_path.touch()

        self.current_path = current_steps_path(self._output_dir, self.env_idx)
        self.step_count = 0
        self.open_current()
        return final_path

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


_VIZ_STEP_KEYS = frozenset({
    "position",
    "reward",
    "action",
    "distance_to_target",
    "terminated",
    "target_position",
    "sim_time",
    "wind",
    "vertical_speed",
})


def load_steps_jsonl(steps_path: Path) -> list[dict[str, Any]]:
    if not steps_path.is_file():
        return []

    steps: list[dict[str, Any]] = []
    with steps_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                steps.append(json.loads(line))
    return steps


def load_viz_steps_jsonl(
    steps_path: Path,
    *,
    max_points: int | None = MAX_VIZ_PLOT_POINTS,
    step_count: int | None = None,
) -> list[dict[str, Any]]:
    """Загрузить поля для 3D-графика; при длинных эпизодах — равномерное прореживание."""
    if not steps_path.is_file():
        return []

    total = step_count
    if max_points is not None and max_points > 0 and total is None:
        total = sum(1 for line in steps_path.open(encoding="utf-8") if line.strip())

    stride = 1
    if max_points is not None and max_points > 0 and total is not None and total > max_points:
        stride = max(1, math.ceil(total / max_points))

    steps: list[dict[str, Any]] = []
    with steps_path.open(encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            if stride > 1 and line_idx % stride != 0:
                continue
            raw = json.loads(line)
            steps.append({key: raw[key] for key in _VIZ_STEP_KEYS if key in raw})
    return steps


def accumulate_position_extents(
    steps_path: Path,
    min_xyz: np.ndarray | None,
    max_xyz: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Обновить min/max координат по JSONL без хранения всех точек в RAM."""
    if not steps_path.is_file():
        return min_xyz, max_xyz

    with steps_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            position = json.loads(line)["position"]
            point = np.asarray(position, dtype=np.float64)
            if min_xyz is None:
                min_xyz = point.copy()
                max_xyz = point.copy()
            else:
                np.minimum(min_xyz, point, out=min_xyz)
                np.maximum(max_xyz, point, out=max_xyz)
    return min_xyz, max_xyz


def include_position_in_extents(
    position: tuple[float, float, float] | list[float],
    min_xyz: np.ndarray | None,
    max_xyz: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    point = np.asarray(position, dtype=np.float64)
    if min_xyz is None:
        return point.copy(), point.copy()
    np.minimum(min_xyz, point, out=min_xyz)
    np.maximum(max_xyz, point, out=max_xyz)
    return min_xyz, max_xyz


def load_last_target_from_jsonl(steps_path: Path) -> list[float] | None:
    """Вернуть target_position последнего шага (один json.loads в конце файла)."""
    if not steps_path.is_file():
        return None

    last_line: str | None = None
    with steps_path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                last_line = stripped
    if last_line is None:
        return None
    return json.loads(last_line).get("target_position")


def cleanup_steps_dir(output_dir: Path) -> None:
    steps_root = steps_dir(output_dir)
    if not steps_root.is_dir():
        return
    for path in steps_root.iterdir():
        path.unlink(missing_ok=True)
    steps_root.rmdir()
