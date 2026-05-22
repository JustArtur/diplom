"""Куррикулум: сужение радиуса успеха по XY (100 m → 25 m)."""

from __future__ import annotations

from dataclasses import dataclass

from stable_baselines3.common.callbacks import BaseCallback

from diplom.envs.constants import TARGET_REACH_RADIUS_FINAL, TARGET_REACH_RADIUS_INITIAL

_LAST_STAGE_UNTIL = 10**12

DEFAULT_TARGET_REACH_CURRICULUM_STAGES: tuple[tuple[int, float], ...] = (
    (15_000_000, TARGET_REACH_RADIUS_INITIAL),
    (_LAST_STAGE_UNTIL, TARGET_REACH_RADIUS_FINAL),
)


@dataclass(frozen=True, slots=True)
class _Stage:
    until_timesteps: int
    target_reach_radius: float


class TrainTargetReachCurriculumCallback(BaseCallback):
    """Обновляет target_reach_radius во всех средах по global timesteps."""

    def __init__(
        self,
        stages: tuple[tuple[int, float], ...] = DEFAULT_TARGET_REACH_CURRICULUM_STAGES,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self._stages = tuple(
            _Stage(int(until), float(radius)) for until, radius in stages
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
        radius = stage.target_reach_radius
        self.training_env.set_attr("target_reach_radius", radius)
        if self.logger is not None:
            self.logger.record("curriculum/target_reach_radius", float(radius))
        if self.verbose:
            print(  # noqa: T201
                f"[target_reach_curriculum] stage {idx + 1}/{len(self._stages)} "
                f"(timesteps<{stage.until_timesteps}): target_reach_radius={radius:.0f} m"
            )
