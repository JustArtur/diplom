from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Callable

from diplom.envs.rewards.types import RewardResult, RewardState, RewardStepContext
from diplom.sim.simulation import SimResult
from diplom.wind.interp import WindInterpolator

RewardFn = Callable[
    [WindInterpolator, SimResult, RewardStepContext, RewardState],
    RewardResult,
]

_PRIVATE_MODULES = frozenset({"__init__", "types"})


def list_reward_names() -> list[str]:
    here = Path(__file__).parent
    return sorted(
        path.stem
        for path in here.glob("*.py")
        if path.stem not in _PRIVATE_MODULES
    )


def get_reward_fn(name: str) -> RewardFn:
    if name not in list_reward_names():
        available = ", ".join(list_reward_names()) or "(пусто)"
        raise ValueError(f"Неизвестная reward-функция {name!r}. Доступные: {available}")
    module = import_module(f"diplom.envs.rewards.{name}")
    compute_reward = getattr(module, "compute_reward", None)
    if compute_reward is None:
        raise ValueError(f"Модуль diplom.envs.rewards.{name} не экспортирует compute_reward")
    return compute_reward


__all__ = [
    "RewardFn",
    "RewardResult",
    "RewardState",
    "RewardStepContext",
    "get_reward_fn",
    "list_reward_names",
]
