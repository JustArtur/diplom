from __future__ import annotations

from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from diplom.viz.trajectory_plot import EpisodeVizData
from diplom.train.trajectory_render_worker import (
    TrajectoryRenderRequest,
    start_trajectory_render_worker,
    stop_trajectory_render_worker,
    submit_trajectory_render,
)


class TrajectoryVisualizationCallback(BaseCallback):
    """Асинхронно рендерит HTML-файлы траекторий в отдельном процессе.

    Args:
        output_dir: каталог для HTML-файлов (создаётся автоматически).
        max_history: сколько завершённых эпизодов показывать на графике.
            Более старые эпизоды бледнее, последний — самый яркий.
        verbose: 0 — тихо, 1 — печатать сводку после каждого rollout.
    """

    def __init__(
        self,
        output_dir: Path,
        max_history: int = 3,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self._output_dir = Path(output_dir)
        self._max_history = max_history

        self._ctx = get_context("spawn")
        self._render_queue = None
        self._render_process = None

        # Шаги текущего (незавершённого) эпизода для каждой среды.
        # Сбрасываются при done=True, накапливаются поперёк rollout-границ.
        self._current_steps: Dict[int, List[dict]] = defaultdict(list)

        # История завершённых эпизодов по env_idx (скользящее окно max_history).
        self._history: Dict[int, List[EpisodeVizData]] = defaultdict(list)

        # Счётчик завершённых эпизодов по env_idx (для метки на графике).
        self._episode_counts: Dict[int, int] = defaultdict(int)

    # ──────────────────── Жизненный цикл ────────────────────

    def _on_training_start(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._render_queue, self._render_process = start_trajectory_render_worker(
                ctx=self._ctx,
                output_dir=self._output_dir,
            )
        except Exception:  # noqa: BLE001
            self._render_queue = None
            self._render_process = None

    def _on_training_end(self) -> None:
        stop_trajectory_render_worker(self._render_queue, self._render_process)
        self._render_queue = None
        self._render_process = None

    # ──────────────────── Накопление шагов ────────────────────

    def _on_step(self) -> bool:
        infos: List[Dict[str, Any]] = self.locals.get("infos", [])
        dones: np.ndarray = self.locals.get("dones", np.zeros(len(infos), dtype=bool))

        for env_idx, (info, done) in enumerate(zip(infos, dones)):
            self._current_steps[env_idx].append(_extract_step(info))
            if done:
                self._finalize_episode(env_idx, info)

        return True

    def _finalize_episode(self, env_idx: int, last_info: Dict[str, Any]) -> None:
        """Переместить накопленные шаги в историю при завершении эпизода."""
        steps = list(self._current_steps[env_idx])
        self._current_steps[env_idx].clear()

        if not steps:
            return

        self._episode_counts[env_idx] += 1
        ep_num = self._episode_counts[env_idx]
        terminated = bool(last_info.get("terminated", False))
        outcome = "успех" if terminated else "truncated"

        episode = EpisodeVizData(
            env_idx=env_idx,
            steps=steps,
            target_position=np.array(
                last_info.get("target_position", [0.0, 0.0, 0.0]), dtype=np.float32
            ),
            label=f"ep {ep_num} ({outcome}, {len(steps)} шагов)",
        )

        history = self._history[env_idx]
        history.append(episode)
        if len(history) > self._max_history:
            history.pop(0)

    # ──────────────────── Асинхронная запись с TensorBoard ────────────────────

    def _on_rollout_end(self) -> None:
        """Отправить последний снапшот в фоновый процесс рендера."""
        if self._render_queue is None:
            return

        payload = TrajectoryRenderRequest(
            num_timesteps=int(self.num_timesteps),
            n_envs=int(getattr(self.training_env, "num_envs", 1)),
            episode_counts=dict(self._episode_counts),
            history={
                env_idx: [_serialize_episode(episode) for episode in episodes]
                for env_idx, episodes in self._history.items()
            },
            current_steps={
                env_idx: [dict(step) for step in steps]
                for env_idx, steps in self._current_steps.items()
            },
        )
        submit_trajectory_render(self._render_queue, payload)


def _serialize_episode(episode: EpisodeVizData) -> dict[str, Any]:
    return {
        "env_idx": int(episode.env_idx),
        "steps": [dict(step) for step in episode.steps],
        "target_position": np.asarray(episode.target_position, dtype=np.float32).tolist(),
        "label": episode.label,
    }


def _extract_step(info: Dict[str, Any]) -> dict:
    return {
        "position": list(np.asarray(info.get("position", [0.0, 0.0, 0.0]), dtype=np.float32)),
        "wind": list(np.asarray(info.get("wind", [0.0, 0.0, 0.0]), dtype=np.float32)),
        "action": float(info.get("action", 0.0)),
        "reward": float(info.get("progress_reward", 0.0)),
        "distance_to_target": float(info.get("distance_to_target", 0.0)),
        "terminated": bool(info.get("terminated", False)),
        "truncated": bool(info.get("truncated", False)),
        "sim_time": str(info.get("sim_time", "")),
        "target_position": list(
            np.asarray(info.get("target_position", [0.0, 0.0, 0.0]), dtype=np.float32)
        ),
    }
