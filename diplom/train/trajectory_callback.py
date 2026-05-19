from __future__ import annotations

from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from diplom.world import WorldBounds
from diplom.train.trajectory_render_worker import (
    TrajectoryRenderRequest,
    cleanup_snapshots_dir,
    snapshot_path_for,
    start_trajectory_render_worker,
    stop_trajectory_render_worker,
    submit_trajectory_render,
    write_trajectory_snapshot,
)
from diplom.train.trajectory_steps_io import (
    EpisodeFileRef,
    EnvStepsWriter,
    cleanup_steps_dir,
)


class TrajectoryVisualizationCallback(BaseCallback):
    """Асинхронно рендерит HTML-файлы траекторий в отдельном процессе.

    Шаги эпизодов пишутся в JSONL на диск по мере симуляции; в RAM хранятся
    только пути к файлам и метаданные для последних ``max_history`` эпизодов.

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

        self._writers: Dict[int, EnvStepsWriter] = {}
        self._history: Dict[int, List[EpisodeFileRef]] = defaultdict(list)
        self._episode_counts: Dict[int, int] = defaultdict(int)
        self._world_bounds: WorldBounds | None = None

    # ──────────────────── Жизненный цикл ────────────────────

    def _on_training_start(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._world_bounds = _get_world_bounds(self.training_env)
        try:
            self._render_queue, self._render_process = start_trajectory_render_worker(
                ctx=self._ctx,
                output_dir=self._output_dir,
            )
        except Exception:  # noqa: BLE001
            self._render_queue = None
            self._render_process = None

    def _on_training_end(self) -> None:
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()

        stop_trajectory_render_worker(self._render_queue, self._render_process)
        self._render_queue = None
        self._render_process = None
        cleanup_snapshots_dir(self._output_dir)
        cleanup_steps_dir(self._output_dir)

    # ──────────────────── Накопление шагов ────────────────────

    def _on_step(self) -> bool:
        dones: np.ndarray = self.locals.get("dones", [])
        step_records: list[dict[str, Any]] = self.training_env.env_method("consume_step_record")

        for env_idx, (record, done) in enumerate(zip(step_records, dones)):
            if not record:
                continue
            self._writer_for(env_idx).append_step(record)
            if done:
                self._finalize_episode(env_idx, record)

        return True

    def _writer_for(self, env_idx: int) -> EnvStepsWriter:
        writer = self._writers.get(env_idx)
        if writer is None:
            writer = EnvStepsWriter(self._output_dir, env_idx)
            writer.open_current()
            self._writers[env_idx] = writer
        return writer

    def _finalize_episode(self, env_idx: int, last_record: Dict[str, Any]) -> None:
        """Закрыть JSONL текущего эпизода и добавить ссылку в историю."""
        writer = self._writers.get(env_idx)
        if writer is None or writer.step_count == 0:
            return

        self._episode_counts[env_idx] += 1
        ep_num = self._episode_counts[env_idx]
        terminated = bool(last_record.get("terminated", False))
        outcome = "успех" if terminated else "truncated"
        step_count = writer.step_count

        steps_path = writer.finalize_episode(ep_num)
        target = _target_position_tuple(last_record)
        episode_ref = EpisodeFileRef(
            steps_path=steps_path,
            env_idx=env_idx,
            target_position=target,
            label=f"ep {ep_num} ({outcome}, {step_count} шагов)",
            step_count=step_count,
        )

        history = self._history[env_idx]
        history.append(episode_ref)
        if len(history) > self._max_history:
            old_ref = history.pop(0)
            old_ref.steps_path.unlink(missing_ok=True)

    # ──────────────────── Асинхронная запись с TensorBoard ────────────────────

    def _on_rollout_end(self) -> None:
        """Записать лёгкий снапшот (пути к JSONL) и передать воркеру."""
        if self._render_queue is None:
            return

        request = self._build_render_request()
        snapshot_path = snapshot_path_for(self._output_dir, request.num_timesteps)
        write_trajectory_snapshot(snapshot_path, request)
        submit_trajectory_render(self._render_queue, snapshot_path)

    def _build_render_request(self) -> TrajectoryRenderRequest:
        current_steps_paths: dict[int, Path] = {}
        current_step_counts: dict[int, int] = {}
        for env_idx, writer in self._writers.items():
            if writer.step_count > 0:
                current_steps_paths[env_idx] = writer.current_path
                current_step_counts[env_idx] = writer.step_count

        return TrajectoryRenderRequest(
            num_timesteps=int(self.num_timesteps),
            n_envs=int(getattr(self.training_env, "num_envs", 1)),
            episode_counts=dict(self._episode_counts),
            history={env_idx: list(episodes) for env_idx, episodes in self._history.items()},
            current_steps_paths=current_steps_paths,
            current_step_counts=current_step_counts,
            world_bounds=self._world_bounds,
        )


def _target_position_tuple(record: Dict[str, Any]) -> tuple[float, float, float]:
    values = record.get("target_position", [0.0, 0.0, 0.0])
    return float(values[0]), float(values[1]), float(values[2])


def _get_world_bounds(training_env) -> WorldBounds | None:
    """Попробовать вытащить общие границы мира из VecEnv."""
    if training_env is None:
        return None

    try:
        bounds_list = training_env.get_attr("world_bounds")
    except Exception:  # noqa: BLE001
        return None

    if not bounds_list:
        return None

    bounds = bounds_list[0]
    return bounds if isinstance(bounds, WorldBounds) else None
