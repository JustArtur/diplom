"""Реестр reward-функций: ``diplom.envs.rewards.<name>`` → ``compute_reward``.

Каждый модуль — самостоятельная reward-модель с полной логикой (без общего _core).
Выбор: ``diplom train-ppo --reward <name>`` или ``diplom rollout --reward <name>``.

Общий контракт
--------------
``compute_reward(wind_interp, step, ctx, state) -> RewardResult``

- ``wind_interp`` — WindInterpolator ERA5; зарезервирован для термов с пробами
  ветра по высоте (пока в большинстве моделей не используется).
- ``step`` — SimResult после ``sim.step()`` (позиция, ветер на текущей Z, цель, …).
- ``ctx`` — RewardStepContext: previous_position, action, energy_delta, boundary,
  лимиты эпизода (см. ``types.py``).
- ``state`` — RewardState: память эпизода между шагами (best distance, z_window, …).

Модуль может экспортировать (опционально, для BalloonEnv):
- ``WIND_ALIGN_SCALE`` — масштаб wind_toward в obs temporal-фичах.
- ``Z_STICK_WINDOW_STEPS`` — длина окна z_stick и maxlen deque в RewardState.

Доступные модели
----------------
simple           — навигация, ветер, энергия; без «мешающих» термов (дефолт)
pbrs             — PBRS по 3D-дистанции + probe-слои; энергия только без Δz
long_horizon     — PBRS + усиленный ветер/scan; слабый штраф за drift (долгий горизонт)
goal_only        — sparse reward только за достижение цели
no_regression    — полный reward без distance_regression
no_z_stick       — полный reward без z_stick (офлайн-симуляции)
weak_z_stick     — полный reward, но мягкий z_stick
"""

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
