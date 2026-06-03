from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diplom.envs.rewards.types import RewardState


@dataclass(frozen=True, slots=True)
class ObsStepContext:

    sim_time: np.datetime64
    z_min: float
    z_max: float
    normalize: bool
    reward_state: RewardState
    wind_align_scale: float
    probe_winds: np.ndarray | None = None
