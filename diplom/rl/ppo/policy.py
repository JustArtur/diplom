"""Общие гиперпараметры MlpPolicy для main-процесса и worker rollout."""

from __future__ import annotations

PPO_NET_ARCH: dict[str, list[int]] = {"pi": [128, 128], "vf": [128, 128]}
# σ ≈ exp(-1.5) ≈ 0.22 при action_limit=5; дальше держим ClipLogStdCallback.
PPO_LOG_STD_INIT: float = -1.5
PPO_LOG_STD_MIN: float = -1.609  # exp ≈ 0.20
PPO_LOG_STD_MAX: float = -0.693  # exp ≈ 0.50


def build_ppo_policy_kwargs() -> dict:
    return {
        "net_arch": PPO_NET_ARCH,
        "log_std_init": PPO_LOG_STD_INIT,
    }


def clamp_policy_log_std(model: object) -> None:
    """Ограничить learnable log_std политики (σ в [PPO_LOG_STD_MIN, PPO_LOG_STD_MAX])."""
    import torch

    policy = getattr(model, "policy", None)
    if policy is None or not hasattr(policy, "log_std"):
        return
    with torch.no_grad():
        policy.log_std.clamp_(PPO_LOG_STD_MIN, PPO_LOG_STD_MAX)
