"""Линейное снижение ent_coef PPO в начале обучения."""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class EntCoefScheduleCallback(BaseCallback):
    """ent_coef: start → end за decay_timesteps (linear)."""

    def __init__(
        self,
        *,
        start: float,
        end: float,
        decay_timesteps: int,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self._start = float(start)
        self._end = float(end)
        self._decay_timesteps = max(1, int(decay_timesteps))

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / self._decay_timesteps)
        ent_coef = self._start + (self._end - self._start) * progress
        self.model.ent_coef = ent_coef
        if self.logger is not None:
            self.logger.record("train/ent_coef", float(ent_coef))
        return True
