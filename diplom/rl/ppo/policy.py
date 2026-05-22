"""Общие хелперы PPO-политики (делегируют в diplom.rl.ppo.models)."""

from __future__ import annotations

from typing import Any

from diplom.rl.ppo.models import get_model_spec


def build_ppo_policy_kwargs(model_name: str = "default") -> dict[str, Any]:
    return get_model_spec(model_name).build_policy_kwargs()


def clamp_policy_log_std(
    model: object,
    *,
    log_std_min: float | None = None,
    log_std_max: float | None = None,
    model_name: str = "default",
) -> None:
    """Ограничить learnable log_std политики."""
    import torch

    spec = get_model_spec(model_name)
    lo = spec.log_std_min if log_std_min is None else log_std_min
    hi = spec.log_std_max if log_std_max is None else log_std_max

    policy = getattr(model, "policy", None)
    if policy is None or not hasattr(policy, "log_std"):
        return
    with torch.no_grad():
        policy.log_std.clamp_(lo, hi)
