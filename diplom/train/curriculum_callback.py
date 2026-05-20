"""Куррикулум: постепенное расширение окна рандомизации старта и цели."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

# (until_timesteps, initial_delta, target_delta) — этап активен, пока num_timesteps < until.
DEFAULT_CURRICULUM_STAGES: tuple[tuple[int, np.ndarray, np.ndarray], ...] = (
    (
        1_000_000,
        np.array([20_000.0, 20_000.0, 5_000.0], dtype=np.float32),
        np.array([20_000.0, 20_000.0, 5_000.0], dtype=np.float32),
    ),
    (
        2_500_000,
        np.array([40_000.0, 40_000.0, 10_000.0], dtype=np.float32),
        np.array([40_000.0, 40_000.0, 10_000.0], dtype=np.float32),
    ),
    (
        10**12,
        np.array([75_000.0, 75_000.0, 5_000.0], dtype=np.float32),
        np.array([75_000.0, 75_000.0, 15_000.0], dtype=np.float32),
    ),
)


@dataclass(frozen=True, slots=True)
class _Stage:
    until_timesteps: int
    initial_delta: np.ndarray
    target_delta: np.ndarray


class TrainPositionCurriculumCallback(BaseCallback):
    """Обновляет train_*_position_delta в средах по числу шагов обучения."""

    def __init__(
        self,
        stages: tuple[tuple[int, np.ndarray, np.ndarray], ...] = DEFAULT_CURRICULUM_STAGES,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self._stages = tuple(
            _Stage(int(until), np.asarray(init, dtype=np.float32), np.asarray(tgt, dtype=np.float32))
            for until, init, tgt in stages
        )
        self._active_idx = -1

    def _on_training_start(self) -> None:
        self._apply_stage(0)

    def _on_step(self) -> bool:
        idx = self._stage_index(self.num_timesteps)
        if idx != self._active_idx:
            self._apply_stage(idx)
        return True

    def _stage_index(self, timesteps: int) -> int:
        for idx, stage in enumerate(self._stages):
            if timesteps < stage.until_timesteps:
                return idx
        return len(self._stages) - 1

    def _apply_stage(self, idx: int) -> None:
        stage = self._stages[idx]
        self._active_idx = idx
        init = stage.initial_delta
        tgt = stage.target_delta
        self.training_env.set_attr("train_initial_position_delta", init)
        self.training_env.set_attr("train_target_position_delta", tgt)
        if self.verbose:
            print(  # noqa: T201
                f"[curriculum] stage {idx + 1}/{len(self._stages)} "
                f"(timesteps<{stage.until_timesteps}): "
                f"init_delta={init.tolist()} target_delta={tgt.tolist()}"
            )
