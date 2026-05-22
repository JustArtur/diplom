"""Общие типы для obs-модулей.

ObsStepContext — общий контекст шага для всех obs-моделей.

Поля
----
- ``sim_time`` — модельное время ERA5 для ``wind_interp.vector_at``.
- ``z_min`` / ``z_max`` — клип probe-высот по границам датасета.
- ``normalize`` — делить компоненты на масштабы из ``envs/constants.py``.
- ``reward_state`` — RewardState эпизода (best distance, adverse_wind_steps,
  last_wind_align_delta для nav/temporal фич; общий объект с reward-модулю).
- ``wind_align_scale`` — из reward-модуля (``WIND_ALIGN_SCALE``), делитель
  для wind_toward и probe winds в obs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diplom.envs.rewards.types import RewardState


@dataclass(frozen=True, slots=True)
class ObsStepContext:
    """Immutable контекст одного шага; поля — см. docstring модуля."""

    sim_time: np.datetime64
    z_min: float
    z_max: float
    normalize: bool
    reward_state: RewardState
    wind_align_scale: float
