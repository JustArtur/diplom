"""Общие гиперпараметры MlpPolicy для main-процесса и worker rollout."""

from __future__ import annotations

PPO_NET_ARCH: dict[str, list[int]] = {"pi": [64, 64], "vf": [64, 64]}
# σ ≈ exp(-1) ≈ 0.37 при action_limit=5 — устойчивый старт без раздувания log_std.
PPO_LOG_STD_INIT: float = -1.0


def build_ppo_policy_kwargs() -> dict:
    return {
        "net_arch": PPO_NET_ARCH,
        "log_std_init": PPO_LOG_STD_INIT,
    }
