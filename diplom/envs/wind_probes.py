# Probe-ветер по высотам: один batch-запрос к WindInterpolator на шаг.

from __future__ import annotations

import numpy as np

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
PROBE_LAYER_GRADIENT_ABOVE_IDX = 4  # +200 m
PROBE_LAYER_GRADIENT_BELOW_IDX = 1  # -200 m
PROBE_WINDS_DZ_THRESHOLD_M = 5.0

_OFFSETS = np.asarray(PROBE_ALTITUDE_OFFSETS_M, dtype=np.float64)


def wind_toward(target: np.ndarray, position: np.ndarray, wind: np.ndarray) -> float:
    delta_xy = target[:2] - position[:2]
    norm = float(np.linalg.norm(delta_xy))
    if norm < 1e-6:
        return 0.0
    unit = delta_xy / norm
    return float(wind[0] * unit[0] + wind[1] * unit[1])


def layer_gradient_from_probe_winds(probe_winds: np.ndarray) -> float:
    return float(
        probe_winds[PROBE_LAYER_GRADIENT_ABOVE_IDX] - probe_winds[PROBE_LAYER_GRADIENT_BELOW_IDX]
    )


def should_compute_probe_winds(
    *,
    obs_needs_probes: bool,
    reward_needs_probes: bool,
    previous_position: np.ndarray | None,
    current_position: np.ndarray,
) -> bool:
    if obs_needs_probes:
        return True
    if not reward_needs_probes:
        return False
    if previous_position is None:
        return False
    dz = abs(float(current_position[2] - previous_position[2]))
    return dz >= PROBE_WINDS_DZ_THRESHOLD_M


def compute_probe_winds(
    wind_interp: WindInterpolator,
    *,
    position: np.ndarray,
    target: np.ndarray,
    sim_time: np.datetime64,
    z_min: float,
    z_max: float,
) -> tuple[np.ndarray, float]:
    # Вернуть (probe_winds[8], max_probe) одним batch_vector_at.
    n = len(_OFFSETS)
    z_probes = np.clip(float(position[2]) + _OFFSETS, z_min, z_max).astype(np.float64)
    x = np.full(n, float(position[0]), dtype=np.float64)
    y = np.full(n, float(position[1]), dtype=np.float64)
    times = np.full(n, sim_time, dtype="datetime64[ns]")

    winds = wind_interp.batch_vector_at(x, y, z_probes, times)
    delta_xy = target[:2].astype(np.float64) - position[:2].astype(np.float64)
    norm = float(np.linalg.norm(delta_xy))
    if norm < 1e-6:
        probe_winds = np.zeros(n, dtype=np.float32)
    else:
        unit = delta_xy / norm
        probe_winds = (winds[:, 0] * unit[0] + winds[:, 1] * unit[1]).astype(np.float32)

    max_probe = float(np.max(probe_winds)) if n else 0.0
    return probe_winds, max_probe
