"""Callback: не даёт PPO раздувать log_std политики выше целевого диапазона σ."""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback

from diplom.rl.ppo.policy import clamp_policy_log_std


class ClipLogStdCallback(BaseCallback):
    """Клампит learnable log_std после каждого rollout."""

    def _on_training_start(self) -> None:
        clamp_policy_log_std(self.model)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> bool:
        clamp_policy_log_std(self.model)
        return True
