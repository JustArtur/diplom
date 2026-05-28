from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from diplom.trajectory.steps_io import list_success_episodes, load_steps_jsonl


@dataclass(frozen=True, slots=True)
class DemoDataset:
    observations: np.ndarray
    actions: np.ndarray
    episode_starts: np.ndarray
    episode_ends: np.ndarray
    episode_ids: np.ndarray

    @property
    def sample_count(self) -> int:
        return int(self.observations.shape[0])

    @property
    def obs_dim(self) -> int:
        return int(self.observations.shape[1]) if self.observations.ndim == 2 else 0

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1]) if self.actions.ndim == 2 else 0


@dataclass(frozen=True, slots=True)
class DemoExportSummary:
    source_dir: Path
    output_path: Path
    summary_path: Path
    episode_count: int
    transition_count: int
    obs_dim: int
    action_dim: int
    episode_lengths: tuple[int, ...]
    episode_files: tuple[str, ...]


def _normalize_demo_root(root: Path) -> Path:
    root = Path(root)
    if root.is_dir() and ((root / "_success").is_dir() or (root / "_steps").is_dir()):
        return root
    trajectories_dir = root / "trajectories"
    if trajectories_dir.is_dir() and ((trajectories_dir / "_success").is_dir() or (trajectories_dir / "_steps").is_dir()):
        return trajectories_dir
    return root


def list_demo_episodes(source_dir: Path) -> list[Path]:
    root = _normalize_demo_root(source_dir)
    if (root / "_success").is_dir():
        return list_success_episodes(root)
    candidates = [path for path in root.rglob("env_*_ep_*.jsonl") if path.parent.name == "_success"]
    return sorted(candidates)


def _step_to_observation(step: dict[str, Any], *, steps_path: Path, step_idx: int) -> np.ndarray:
    if "observation" not in step:
        raise ValueError(
            f"В {steps_path} не найдено поле 'observation' для шага {step_idx + 1}. "
            "Для pretraining нужны демонстрации, записанные с trajectory_record_observation=True."
        )
    return np.asarray(step["observation"], dtype=np.float32)


def _step_to_action(step: dict[str, Any]) -> np.ndarray:
    action = np.asarray(step["action"], dtype=np.float32)
    return np.atleast_1d(action).astype(np.float32, copy=False)


def _load_demo_episode(steps_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    steps = load_steps_jsonl(steps_path)
    if not steps:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=bool),
            np.empty((0,), dtype=bool),
        )

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for step_idx, step in enumerate(steps):
        observations.append(_step_to_observation(step, steps_path=steps_path, step_idx=step_idx))
        actions.append(_step_to_action(step))

    obs_arr = np.stack(observations, axis=0).astype(np.float32, copy=False)
    action_arr = np.stack(actions, axis=0).astype(np.float32, copy=False)
    episode_starts = np.zeros((obs_arr.shape[0],), dtype=bool)
    episode_starts[0] = True
    episode_ends = np.zeros((obs_arr.shape[0],), dtype=bool)
    episode_ends[-1] = True
    return obs_arr, action_arr, episode_starts, episode_ends


def load_demo_dataset(dataset_path: Path) -> DemoDataset:
    dataset_path = Path(dataset_path)
    with np.load(dataset_path, allow_pickle=False) as data:
        observations = np.asarray(data["observations"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        episode_starts = np.asarray(data["episode_starts"], dtype=bool)
        episode_ends = np.asarray(data["episode_ends"], dtype=bool)
        episode_ids = np.asarray(data["episode_ids"], dtype=np.int32)
    return DemoDataset(
        observations=observations,
        actions=actions,
        episode_starts=episode_starts,
        episode_ends=episode_ends,
        episode_ids=episode_ids,
    )


def export_demo_dataset(
    source_dir: Path,
    output_path: Path,
    *,
    max_episodes: int | None = None,
) -> DemoExportSummary:
    root = _normalize_demo_root(source_dir)
    episode_paths = list_demo_episodes(root)
    if max_episodes is not None:
        episode_paths = episode_paths[:max_episodes]
    if not episode_paths:
        raise ValueError(
            f"В {root} не найдено успешных эпизодов. "
            "Сначала пройдите хотя бы один эпизод в интерактивном режиме."
        )

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    episode_starts: list[np.ndarray] = []
    episode_ends: list[np.ndarray] = []
    episode_ids: list[np.ndarray] = []
    episode_lengths: list[int] = []
    episode_files: list[str] = []

    for episode_idx, steps_path in enumerate(episode_paths):
        obs_arr, action_arr, starts_arr, ends_arr = _load_demo_episode(steps_path)
        if obs_arr.size == 0:
            continue
        episode_len = int(obs_arr.shape[0])
        observations.append(obs_arr)
        actions.append(action_arr)
        episode_starts.append(starts_arr)
        episode_ends.append(ends_arr)
        episode_ids.append(np.full((episode_len,), episode_idx, dtype=np.int32))
        episode_lengths.append(episode_len)
        episode_files.append(str(steps_path.resolve()))

    if not observations:
        raise ValueError(
            f"В {root} не нашлось непустых демонстраций для экспорта."
        )

    observations_arr = np.concatenate(observations, axis=0)
    actions_arr = np.concatenate(actions, axis=0)
    episode_starts_arr = np.concatenate(episode_starts, axis=0)
    episode_ends_arr = np.concatenate(episode_ends, axis=0)
    episode_ids_arr = np.concatenate(episode_ids, axis=0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        observations=observations_arr,
        actions=actions_arr,
        episode_starts=episode_starts_arr,
        episode_ends=episode_ends_arr,
        episode_ids=episode_ids_arr,
    )

    summary_path = output_path.with_suffix(".json")
    summary = {
        "source_dir": str(root.resolve()),
        "output_path": str(output_path.resolve()),
        "episode_count": len(episode_lengths),
        "transition_count": int(observations_arr.shape[0]),
        "obs_dim": int(observations_arr.shape[1]),
        "action_dim": int(actions_arr.shape[1]),
        "episode_lengths": episode_lengths,
        "episode_files": episode_files,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    return DemoExportSummary(
        source_dir=root.resolve(),
        output_path=output_path.resolve(),
        summary_path=summary_path.resolve(),
        episode_count=len(episode_lengths),
        transition_count=int(observations_arr.shape[0]),
        obs_dim=int(observations_arr.shape[1]),
        action_dim=int(actions_arr.shape[1]),
        episode_lengths=tuple(episode_lengths),
        episode_files=tuple(episode_files),
    )
