from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class EpisodeStatsCallback(BaseCallback):
    # Пишет rollout/ep_rew_mean и custom/success_rate при завершении эпизодов

    def __init__(self) -> None:
        super().__init__(verbose=0)
        self._ep_returns: list[float] = []
        self._ep_lengths: list[float] = []
        self._successes: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if not infos:
            return True
        for info in infos:
            if not info:
                continue
            episode = info.get("episode")
            if episode is not None:
                self._ep_returns.append(float(episode["r"]))
                self._ep_lengths.append(float(episode["l"]))
                self._successes.append(1.0 if info.get("terminated") else 0.0)
        return True

    def _on_rollout_end(self) -> None:
        if not self.logger:
            return
        if self._ep_returns:
            n = len(self._ep_returns)
            self.logger.record("rollout/ep_rew_mean", sum(self._ep_returns) / n)
            self.logger.record("rollout/ep_len_mean", sum(self._ep_lengths) / n)
            self._ep_returns.clear()
            self._ep_lengths.clear()
        if self._successes:
            n = len(self._successes)
            self.logger.record("custom/success_rate", sum(self._successes) / n)
            self._successes.clear()
