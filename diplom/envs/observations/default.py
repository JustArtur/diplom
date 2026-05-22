"""Obs ``default`` — полный вектор наблюдений с probe-ветром по высотам.

CLI: ``--obs default``   OBS_DIM = 33

Структура вектора (индексы)
---------------------------
[0:3]   position (x, y, z)
[3:6]   target_position
[6:9]   delta = target - position
[9:12]  wind (u, v, w) на текущей высоте баллона
[12]    energy_spent
[13]    air_weight
[14]    vertical_speed
[15]    vertical_acceleration
[16]    air_density
[17]    temperature
[18]    pressure
[19]    wind_toward — проекция ветра на направление к цели (текущая Z)
[20:28] probe_winds — wind_toward на 8 высотах (см. PROBE_ALTITUDE_OFFSETS_M)
[28:30] nav: [dist_xy / 50km, best_horizontal / curr_horizontal]
[30:33] temporal: [adverse_wind_steps/1k, Δwind_toward/scale, layer_grad/scale]

Probe-ветер
-----------
``PROBE_ALTITUDE_OFFSETS_M``: -400, -200, -100, +100, +200, +400, +1000, +2500 м
от текущей Z. Для каждого offset:
  z_probe = clip(z + offset, z_min, z_max)
  wind = wind_interp.vector_at(x, y, z_probe, sim_time)
  probe_winds[i] = dot(wind_xy, unit_to_target)

Layer gradient (temporal[2])
--------------------------
(wind_toward@Z+200m) - (wind_toward@Z-200m) — вертикальный градиент «выгодности» ветра.

Нормализация
------------
При ``ctx.normalize=True`` XYZ делятся на OBS_XY_SCALE / OBS_ALTITUDE_SCALE,
ветер на OBS_WIND_SCALE, wind_toward и probe_winds на wind_align_scale,
остальные скаляры — на свои OBS_*_SCALE из constants.py.

Зависимость от reward
---------------------
nav и temporal читают ``ctx.reward_state`` — obs и reward делят RewardState
в BalloonEnv (best distance, adverse steps, wind delta).
"""

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

PROBE_ALTITUDE_OFFSETS_M: tuple[float, ...] = (
    -400.0,
    -200.0,
    -100.0,
    100.0,
    200.0,
    400.0,
    1_000.0,
    2_500.0,
)
PROBE_LAYER_GRADIENT_DELTA_M = 200.0
# base(20) + probe winds(N) + nav(2) + temporal(3)
OBS_DIM = 20 + len(PROBE_ALTITUDE_OFFSETS_M) + 2 + 3


def _horizontal_distance(target: np.ndarray, position: np.ndarray) -> float:
    return float(np.linalg.norm((target[:2] - position[:2]).astype(np.float64)))


def _wind_toward(target: np.ndarray, position: np.ndarray, wind: np.ndarray) -> float:
    delta_xy = target[:2] - position[:2]
    norm = float(np.linalg.norm(delta_xy))
    if norm < 1e-6:
        return 0.0
    unit = delta_xy / norm
    return float(wind[0] * unit[0] + wind[1] * unit[1])


def _clip_probe_altitude(z_probe_m: float, *, z_min: float, z_max: float) -> float:
    return float(np.clip(z_probe_m, z_min, z_max))


def _probe_wind_toward(
    wind_interp: WindInterpolator,
    *,
    position: np.ndarray,
    target: np.ndarray,
    z_probe_m: float,
    sim_time: np.datetime64,
) -> float:
    sample = wind_interp.vector_at(
        float(position[0]),
        float(position[1]),
        float(z_probe_m),
        sim_time,
    )
    wind = np.array([sample.u, sample.v, sample.w], dtype=np.float32)
    return _wind_toward(target, position, wind)


def _probe_wind_toward_at_offset(
    wind_interp: WindInterpolator,
    *,
    position: np.ndarray,
    target: np.ndarray,
    z_offset_m: float,
    ctx: ObsStepContext,
) -> float:
    z_probe = _clip_probe_altitude(
        float(position[2]) + z_offset_m,
        z_min=ctx.z_min,
        z_max=ctx.z_max,
    )
    return _probe_wind_toward(
        wind_interp,
        position=position,
        target=target,
        z_probe_m=z_probe,
        sim_time=ctx.sim_time,
    )


def _probe_winds_around_altitude(
    wind_interp: WindInterpolator,
    *,
    position: np.ndarray,
    target: np.ndarray,
    ctx: ObsStepContext,
) -> np.ndarray:
    return np.array(
        [
            _probe_wind_toward_at_offset(
                wind_interp,
                position=position,
                target=target,
                z_offset_m=offset,
                ctx=ctx,
            )
            for offset in PROBE_ALTITUDE_OFFSETS_M
        ],
        dtype=np.float32,
    )


def _wind_layer_gradient(
    wind_interp: WindInterpolator,
    *,
    position: np.ndarray,
    target: np.ndarray,
    ctx: ObsStepContext,
) -> float:
    delta = PROBE_LAYER_GRADIENT_DELTA_M
    wind_above = _probe_wind_toward_at_offset(
        wind_interp,
        position=position,
        target=target,
        z_offset_m=delta,
        ctx=ctx,
    )
    wind_below = _probe_wind_toward_at_offset(
        wind_interp,
        position=position,
        target=target,
        z_offset_m=-delta,
        ctx=ctx,
    )
    return wind_above - wind_below


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
    position = np.asarray(step.position, dtype=np.float32)
    target = np.asarray(step.target_position, dtype=np.float32)
    delta = target - position
    wind = np.asarray(step.wind, dtype=np.float32)
    wind_toward = _wind_toward(target, position, wind)
    curr_horizontal = _horizontal_distance(target, position)
    best_ratio = ctx.reward_state.best_horizontal_distance / max(curr_horizontal, 1.0)
    probe_winds = _probe_winds_around_altitude(
        wind_interp,
        position=position,
        target=target,
        ctx=ctx,
    )
    nav_features = np.array(
        [
            curr_horizontal / OBS_NAV_DISTANCE_SCALE,
            best_ratio,
        ],
        dtype=np.float32,
    )
    wind_layer_gradient = _wind_layer_gradient(
        wind_interp,
        position=position,
        target=target,
        ctx=ctx,
    )
    temporal_features = np.array(
        [
            ctx.reward_state.adverse_wind_steps / OBS_ADVERSE_WIND_STEPS_SCALE,
            ctx.reward_state.last_wind_align_delta / ctx.wind_align_scale,
            wind_layer_gradient / ctx.wind_align_scale,
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
                probe_winds,
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
            probe_winds / ctx.wind_align_scale,
            nav_features,
            temporal_features,
        ],
        dtype=np.float32,
    )
