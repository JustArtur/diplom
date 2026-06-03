# PPO default: MlpPolicy 128-128.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

POLICY_TYPE = "MlpPolicy"
NET_ARCH: dict[str, list[int]] = {"pi": [128, 128], "vf": [128, 128]}
LOG_STD_INIT = -1.5
LOG_STD_MIN = -1.609  # exp ≈ 0.20
LOG_STD_MAX = -0.693  # exp ≈ 0.50


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    policy_type: str
    net_arch: dict[str, list[int]]
    log_std_init: float
    log_std_min: float
    log_std_max: float
    recurrent: bool = False
    lstm_hidden_size: int = 256
    n_lstm_layers: int = 1
    enable_critic_lstm: bool = True
    shared_lstm: bool = False

    def build_policy_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"net_arch": self.net_arch, "log_std_init": self.log_std_init}
        if self.recurrent:
            kwargs.update(
                lstm_hidden_size=self.lstm_hidden_size,
                n_lstm_layers=self.n_lstm_layers,
                enable_critic_lstm=self.enable_critic_lstm,
                shared_lstm=self.shared_lstm,
            )
        return kwargs


SPEC = ModelSpec(
    name="default",
    policy_type=POLICY_TYPE,
    net_arch=NET_ARCH,
    log_std_init=LOG_STD_INIT,
    log_std_min=LOG_STD_MIN,
    log_std_max=LOG_STD_MAX,
)
