from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Максимум точек на одну траекторию в HTML (Plotly тормозит на сотнях тысяч).
MAX_VIZ_PLOT_POINTS = 8_000


STEPS_SUBDIR = "_steps"
SUCCESS_STEPS_SUBDIR = "_success"
CURRENT_STEPS_FILENAME = "env_{env_idx:03d}_current.jsonl"
EPISODE_STEPS_FILENAME = "env_{env_idx:03d}_ep_{episode_num:06d}.jsonl"
EPISODE_META_SUFFIX = ".meta.json"


def steps_dir(output_dir: Path) -> Path:
    return Path(output_dir) / STEPS_SUBDIR


def success_steps_dir(output_dir: Path) -> Path:
    return Path(output_dir) / SUCCESS_STEPS_SUBDIR


def current_steps_path(output_dir: Path, env_idx: int) -> Path:
    return steps_dir(output_dir) / CURRENT_STEPS_FILENAME.format(env_idx=env_idx)


def episode_steps_path(output_dir: Path, env_idx: int, episode_num: int) -> Path:
    return steps_dir(output_dir) / EPISODE_STEPS_FILENAME.format(
        env_idx=env_idx,
        episode_num=episode_num,
    )


def success_episode_steps_path(output_dir: Path, env_idx: int, episode_num: int) -> Path:
    return success_steps_dir(output_dir) / EPISODE_STEPS_FILENAME.format(
        env_idx=env_idx,
        episode_num=episode_num,
    )


def success_episode_meta_path(output_dir: Path, env_idx: int, episode_num: int) -> Path:
    return success_episode_steps_path(output_dir, env_idx, episode_num).with_suffix(
        EPISODE_META_SUFFIX,
    )


@dataclass(frozen=True, slots=True)
class EpisodeFileRef:
    steps_path: Path
    env_idx: int
    target_position: tuple[float, float, float]
    label: str
    step_count: int
    success: bool = False


class EnvStepsWriter:
    """Пишет шаги эпизода в JSONL на диск, не держа их в RAM."""

    def __init__(self, output_dir: Path, env_idx: int) -> None:
        self._output_dir = Path(output_dir)
        self.output_dir = self._output_dir
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

    def flush(self) -> None:
        """Сбросить буфер JSONL на диск перед чтением снапшотом live-рендера."""
        if self._handle is not None:
            self._handle.flush()

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


def _read_jsonl_first_record(steps_path: Path) -> dict[str, Any] | None:
    if not steps_path.is_file():
        return None
    with steps_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                return json.loads(line)
    return None


def _read_jsonl_last_record(steps_path: Path) -> dict[str, Any] | None:
    if not steps_path.is_file():
        return None

    with steps_path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        if size == 0:
            return None
        read_size = min(size, 65_536)
        handle.seek(-read_size, 2)
        chunk = handle.read(read_size)

    for raw_line in reversed(chunk.splitlines()):
        line = raw_line.decode("utf-8", errors="ignore").strip()
        if line:
            return json.loads(line)
    return None


def archive_success_episode(
    source_path: Path,
    output_dir: Path,
    *,
    env_idx: int,
    episode_num: int,
    step_count: int | None = None,
) -> Path:
    """Скопировать полный JSONL эпизода в ``_success/`` и записать метаданные для replay."""
    dest_path = success_episode_steps_path(output_dir, env_idx, episode_num)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, dest_path)

    first = _read_jsonl_first_record(dest_path)
    last = _read_jsonl_last_record(dest_path)
    if step_count is None:
        step_count = sum(1 for line in dest_path.open(encoding="utf-8") if line.strip())
    meta_path = success_episode_meta_path(output_dir, env_idx, episode_num)
    meta: dict[str, Any] = {
        "env_idx": env_idx,
        "episode_num": episode_num,
        "step_count": step_count,
        "success": True,
        "steps_file": dest_path.name,
        "actions_field": "action",
    }
    if first is not None:
        meta["initial_position"] = first.get("position")
        meta["target_position"] = first.get("target_position")
        meta["initial_sim_time"] = first.get("sim_time")
        meta["initial_horizontal_distance_m"] = first.get("horizontal_distance")
    if last is not None:
        meta["final_position"] = last.get("position")
        meta["final_sim_time"] = last.get("sim_time")
        meta["final_horizontal_distance_m"] = last.get("horizontal_distance")
        meta["terminated"] = bool(last.get("terminated", False))
        meta["truncated"] = bool(last.get("truncated", False))

    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return dest_path


def list_success_episodes(output_dir: Path) -> list[Path]:
    root = success_steps_dir(output_dir)
    if not root.is_dir():
        return []
    return sorted(root.glob("env_*_ep_*.jsonl"))


def load_success_episode_meta(meta_path: Path) -> dict[str, Any]:
    with meta_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_replay_actions(steps_path: Path) -> list[float]:
    """Действия модели по шагам (для open-loop replay)."""
    return [float(step["action"]) for step in load_steps_jsonl(steps_path)]


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
    *,
    step_count: int | None = None,
    max_samples: int = 4_000,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Обновить min/max координат по JSONL без хранения всех точек в RAM."""
    if not steps_path.is_file():
        return min_xyz, max_xyz

    stride = 1
    if step_count is not None and step_count > max_samples:
        stride = max(1, math.ceil(step_count / max_samples))

    with steps_path.open(encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            if stride > 1 and line_idx % stride != 0:
                continue
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
    """Вернуть target_position последнего шага (читаем хвост файла)."""
    if not steps_path.is_file():
        return None

    with steps_path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        if size == 0:
            return None
        read_size = min(size, 65_536)
        handle.seek(-read_size, 2)
        chunk = handle.read(read_size)

    lines = chunk.splitlines()
    for raw_line in reversed(lines):
        line = raw_line.decode("utf-8", errors="ignore").strip()
        if line:
            return json.loads(line).get("target_position")
    return None


def cleanup_steps_dir(output_dir: Path) -> None:
    """Удалить временные JSONL в ``_steps/``; архив ``_success/`` не трогать."""
    steps_root = steps_dir(output_dir)
    if not steps_root.is_dir():
        return
    for path in steps_root.iterdir():
        path.unlink(missing_ok=True)
    steps_root.rmdir()
