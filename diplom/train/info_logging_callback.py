"""SB3-callback: логирование полей info из каждой среды в TensorBoard."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class InfoLoggingCallback(BaseCallback):
    """Собирает `info` из каждой среды отдельно и пишет агрегаты в логгер PPO.

    PPO сам не выводит содержимое `info`, поэтому мы собираем числовые поля
    за один rollout по каждой среде и логируем их в stdout и TensorBoard
    через стандартный logger.
    """

    _IGNORED_KEYS = {"terminal_observation", "final_observation"}

    def __init__(self, prefix: str = "env") -> None:
        super().__init__(verbose=0)
        self.prefix = prefix
        self._values: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        self._samples: dict[int, int] = defaultdict(int)

    def _on_rollout_start(self) -> None:
        self._values.clear()
        self._samples.clear()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for env_idx, info in enumerate(infos):
            if isinstance(info, dict):
                self._collect_info(env_idx, info)
                self._samples[env_idx] += 1
        return True

    def _on_rollout_end(self) -> None:
        if not self._values:
            return

        for env_idx in sorted(self._values):
            env_prefix = f"{self.prefix}_{env_idx}"
            self.logger.record(f"{env_prefix}/samples", self._samples[env_idx])
            for key, values in sorted(self._values[env_idx].items()):
                self.logger.record(f"{env_prefix}/{key}", self._aggregate_values(key, values))

    def _collect_info(self, env_idx: int, info: dict[str, Any], path: str = "") -> None:
        for key, value in info.items():
            if key in self._IGNORED_KEYS:
                continue
            name = f"{path}{key}"
            if isinstance(value, dict):
                self._collect_info(env_idx, value, f"{name}/")
                continue
            for flat_key, flat_value in self._flatten_value(name, value).items():
                self._values[env_idx][flat_key].append(flat_value)

    def _flatten_value(self, key: str, value: Any) -> dict[str, Any]:
        if isinstance(value, (bool, int, float, np.integer, np.floating)):
            return {key: float(value)}

        array = np.asarray(value)
        if array.ndim == 0 and np.issubdtype(array.dtype, np.number):
            return {key: float(array.item())}

        if array.ndim >= 1 and np.issubdtype(array.dtype, np.number):
            flat = array.reshape(-1)
            return {f"{key}/{idx}": float(item) for idx, item in enumerate(flat)}

        return {}

    def _aggregate_values(self, key: str, values: list[Any]) -> Any:
        return float(np.mean(values))
