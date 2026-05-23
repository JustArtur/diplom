"""Реестр obs-модулей: ``diplom.envs.observations.<name>`` → ``build_obs``.

Каждый модуль — самостоятельная модель наблюдений с собственным ``OBS_DIM``.
Выбор: ``diplom train-ppo --obs <name>`` (должен совпадать с размерностью
входа PPO при resume/load checkpoint).

Общий контракт
--------------
``build_obs(wind_interp, step, ctx) -> np.ndarray`` shape ``(OBS_DIM,)``

- ``wind_interp`` — WindInterpolator; probe-модели опрашивают ветер на
  разных высотах через ``vector_at(x, y, z_probe, sim_time)``.
- ``step`` — SimResult (позиция, ветер на текущей Z, физика, цель).
- ``ctx`` — ObsStepContext: sim_time, границы Z, normalize, RewardState,
  wind_align_scale (см. ``types.py``).

Модуль обязан экспортировать
----------------------------
- ``build_obs`` — функция сборки вектора.
- ``OBS_DIM`` — int, размер ``observation_space``.

Доступные модели
----------------
default  — OBS_DIM=33, probe wind_toward на 8 высотах + nav + temporal
minimal  — OBS_DIM=24, без probe-слоёв и layer gradient
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Callable

import numpy as np

from diplom.envs.observations.types import ObsStepContext
from diplom.sim.simulation import SimResult
from diplom.wind.interp import WindInterpolator

ObsFn = Callable[[WindInterpolator, SimResult, ObsStepContext], np.ndarray]

_PRIVATE_MODULES = frozenset({"__init__", "types"})


def list_obs_names() -> list[str]:
    here = Path(__file__).parent
    return sorted(
        path.stem
        for path in here.glob("*.py")
        if path.stem not in _PRIVATE_MODULES
    )


def get_obs_spec(name: str) -> tuple[ObsFn, int]:
    if name not in list_obs_names():
        available = ", ".join(list_obs_names()) or "(пусто)"
        raise ValueError(f"Неизвестная obs-модель {name!r}. Доступные: {available}")
    module = import_module(f"diplom.envs.observations.{name}")
    build_obs = getattr(module, "build_obs", None)
    obs_dim = getattr(module, "OBS_DIM", None)
    if build_obs is None or obs_dim is None:
        raise ValueError(
            f"Модуль diplom.envs.observations.{name} должен экспортировать build_obs и OBS_DIM"
        )
    return build_obs, int(obs_dim)


__all__ = [
    "ObsFn",
    "ObsStepContext",
    "get_obs_spec",
    "list_obs_names",
]
