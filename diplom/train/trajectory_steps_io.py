from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
        self._handle.flush()
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


def cleanup_steps_dir(output_dir: Path) -> None:
    steps_root = steps_dir(output_dir)
    if not steps_root.is_dir():
        return
    for path in steps_root.iterdir():
        path.unlink(missing_ok=True)
    steps_root.rmdir()
