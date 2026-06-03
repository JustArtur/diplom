# ObsStepContext для сборки наблюдений.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diplom.envs.rewards.types import RewardState


@dataclass(frozen=True, slots=True)
class ObsStepContext:
    # Immutable контекст одного шага; поля, см. docstring модуля.

    sim_time: np.datetime64
    z_min: float
    z_max: float
    normalize: bool
    reward_state: RewardState
    wind_align_scale: float
    probe_winds: np.ndarray | None = None
