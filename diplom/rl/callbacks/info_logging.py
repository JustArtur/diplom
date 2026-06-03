from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback

from diplom.rl.logging.env_info_log_keys import ENV_INFO_LOG_KEYS


class InfoLoggingCallback(BaseCallback):
    # Собирает выбранные скалярные поля info и пишет mean за rollout в TensorBoard
    # PPO сам не выводит содержимое info, поэтому callback накапливает только
    # заранее заданный набор ключей и логирует среднее по шагам rollout
    # В stdout метрики не попадают, только TensorBoard

    _KEYS = ENV_INFO_LOG_KEYS

    def __init__(self, prefix: str = "env") -> None:
        super().__init__(verbose=0)
        self.prefix = prefix
        self._n_envs = 0
        self._samples: list[int] = []
        self._sums: dict[str, list[float]] = {}
        self._counts: dict[str, list[int]] = {}
        self._tag_samples: tuple[str, ...] = ()
        self._tag_keys: dict[str, tuple[str, ...]] = {}

    def _on_training_start(self) -> None:
        n = self.training_env.num_envs
        self._n_envs = n
        p = self.prefix
        self._tag_samples = tuple(f"{p}_{i}/samples" for i in range(n))
        self._tag_keys = {
            key: tuple(f"{p}_{i}/{key}" for i in range(n))
            for key in self._KEYS
        }
        self._reset_buffers()

    def _on_rollout_start(self) -> None:
        self._reset_buffers()

    def _reset_buffers(self) -> None:
        n = self._n_envs
        self._samples = [0] * n
        self._sums = {key: [0.0] * n for key in self._KEYS}
        self._counts = {key: [0] * n for key in self._KEYS}

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        samples = self._samples
        sums = self._sums
        counts = self._counts
        for env_idx, info in enumerate(infos):
            if not info:
                continue
            samples[env_idx] += 1
            for key in self._KEYS:
                val = info.get(key)
                if val is not None:
                    sums[key][env_idx] += val
                    counts[key][env_idx] += 1
        return True

    def _on_rollout_end(self) -> None:
        logger = self.logger
        tag_samples = self._tag_samples
        tag_keys = self._tag_keys
        samples = self._samples
        sums = self._sums
        counts = self._counts
        for env_idx in range(self._n_envs):
            n_samples = samples[env_idx]
            if not n_samples:
                continue
            logger.record(tag_samples[env_idx], n_samples, exclude="stdout")
            for key in self._KEYS:
                c = counts[key][env_idx]
                if c:
                    logger.record(
                        tag_keys[key][env_idx],
                        sums[key][env_idx] / c,
                        exclude="stdout",
                    )
