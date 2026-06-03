from __future__ import annotations

import numpy as np

from diplom.envs.constants import (
    OBS_ADVERSE_WIND_STEPS_SCALE,
    OBS_AIR_DENSITY_SCALE,
    OBS_AIR_WEIGHT_SCALE,
    OBS_ALTITUDE_SCALE,
    OBS_ENERGY_SCALE,
    OBS_NAV_DISTANCE_SCALE,
    OBS_PRESSURE_SCALE,
    OBS_TEMPERATURE_SCALE,
    OBS_VERTICAL_ACCELERATION_SCALE,
    OBS_VERTICAL_SPEED_SCALE,
    OBS_WIND_SCALE,
    OBS_XY_SCALE,
)
from diplom.envs.observations.types import ObsStepContext
from diplom.sim.simulation import SimResult
from diplom.wind.interp import WindInterpolator

OBS_DIM = 20 + 2 + 2  # base(20, incl. wind_toward) + nav + temporal (без layer gradient)


def _horizontal_distance(target: np.ndarray, position: np.ndarray) -> float:
    return float(np.linalg.norm((target[:2] - position[:2]).astype(np.float64)))


def _wind_toward(target: np.ndarray, position: np.ndarray, wind: np.ndarray) -> float:
    delta_xy = target[:2] - position[:2]
    norm = float(np.linalg.norm(delta_xy))
    if norm < 1e-6:
        return 0.0
    unit = delta_xy / norm
    return float(wind[0] * unit[0] + wind[1] * unit[1])


def _scale_xyz(vec: np.ndarray, xy_scale: float, z_scale: float) -> np.ndarray:
    out = vec.astype(np.float32, copy=True)
    out[0] /= xy_scale
    out[1] /= xy_scale
    out[2] /= z_scale
    return out


def build_obs(
    wind_interp: WindInterpolator,
    step: SimResult,
    ctx: ObsStepContext,
) -> np.ndarray:
    del wind_interp

    position = np.asarray(step.position, dtype=np.float32)
    target = np.asarray(step.target_position, dtype=np.float32)
    delta = target - position
    wind = np.asarray(step.wind, dtype=np.float32)
    wind_toward = _wind_toward(target, position, wind)
    curr_horizontal = _horizontal_distance(target, position)
    best_ratio = ctx.reward_state.best_horizontal_distance / max(curr_horizontal, 1.0)
    nav_features = np.array(
        [
            curr_horizontal / OBS_NAV_DISTANCE_SCALE,
            best_ratio,
        ],
        dtype=np.float32,
    )
    temporal_features = np.array(
        [
            ctx.reward_state.adverse_wind_steps / OBS_ADVERSE_WIND_STEPS_SCALE,
            ctx.reward_state.last_wind_align_delta / ctx.wind_align_scale,
        ],
        dtype=np.float32,
    )

    if not ctx.normalize:
        return np.concatenate(
            [
                position,
                target,
                delta,
                wind,
                [step.energy_spent],
                [step.air_weight],
                [step.vertical_speed],
                [step.vertical_acceleration],
                [step.air_density],
                [step.temperature],
                [step.pressure],
                [wind_toward],
                nav_features,
                temporal_features,
            ],
            dtype=np.float32,
        )

    return np.concatenate(
        [
            _scale_xyz(position, OBS_XY_SCALE, OBS_ALTITUDE_SCALE),
            _scale_xyz(target, OBS_XY_SCALE, OBS_ALTITUDE_SCALE),
            _scale_xyz(delta, OBS_XY_SCALE, OBS_ALTITUDE_SCALE),
            wind / OBS_WIND_SCALE,
            [step.energy_spent / OBS_ENERGY_SCALE],
            [step.air_weight / OBS_AIR_WEIGHT_SCALE],
            [step.vertical_speed / OBS_VERTICAL_SPEED_SCALE],
            [step.vertical_acceleration / OBS_VERTICAL_ACCELERATION_SCALE],
            [step.air_density / OBS_AIR_DENSITY_SCALE],
            [step.temperature / OBS_TEMPERATURE_SCALE],
            [step.pressure / OBS_PRESSURE_SCALE],
            [wind_toward / ctx.wind_align_scale],
            nav_features,
            temporal_features,
        ],
        dtype=np.float32,
    )
