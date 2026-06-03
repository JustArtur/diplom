# Reward pbrs: PBRS по 3D-дистанции и probe-ветру.

from __future__ import annotations

import numpy as np

from diplom.envs.rewards.types import RewardResult, RewardState, RewardStepContext
from diplom.sim.simulation import SimResult
from diplom.wind.interp import WindInterpolator

NEEDS_PROBE_WINDS = True
WIND_ALIGN_SCALE = 20.0
WIND_ALIGN_COEF = 0.5
WIND_ALIGN_DELTA_COEF = 0.25
WIND_SCAN_MIN_DZ_M = 5.0
WIND_SCAN_DELTA_COEF = 2.0
WIND_SCAN_MAX_DIST_M = 50_000.0
PROBE_LAYER_MIN_DZ_M = 5.0
PROBE_LAYER_COEF = 1.5
ENERGY_COEF = 0.5
ENERGY_SCALE = 100.0
IDLE_PUMP_MIN_DZ_M = 5.0
BOUNDARY_PENALTY = 0.05
PBRS_GAMMA = 0.99
PBRS_COEF = 1.0
PBRS_DISTANCE_SCALE = 75_000.0
SUCCESS_REWARD = 500.0
Z_STICK_WINDOW_STEPS = 1


def _horizontal_distance(target: np.ndarray, position: np.ndarray) -> float:
    return float(np.linalg.norm((target[:2] - position[:2]).astype(np.float64)))


def _vertical_distance(target: np.ndarray, position: np.ndarray) -> float:
    return float(abs(float(target[2]) - float(position[2])))


def _distance_3d(target: np.ndarray, position: np.ndarray) -> float:
    return float(np.linalg.norm((target - position).astype(np.float64)))


def _wind_toward(target: np.ndarray, position: np.ndarray, wind: np.ndarray) -> float:
    delta_xy = target[:2] - position[:2]
    norm = float(np.linalg.norm(delta_xy))
    if norm < 1e-6:
        return 0.0
    unit = delta_xy / norm
    return float(wind[0] * unit[0] + wind[1] * unit[1])


def _pbrs_term(prev_distance: float, curr_distance: float) -> float:
    delta = prev_distance - PBRS_GAMMA * curr_distance
    return PBRS_COEF * delta / PBRS_DISTANCE_SCALE


def _energy_term(energy_delta: float, dz: float) -> float:
    if dz >= IDLE_PUMP_MIN_DZ_M:
        return 0.0
    return -ENERGY_COEF * energy_delta / ENERGY_SCALE


def _probe_layer_term(
    *,
    climbing: bool,
    max_probe: float,
    wind_toward: float,
) -> float:
    if not climbing:
        return 0.0
    gain = max(0.0, max_probe - wind_toward)
    return PROBE_LAYER_COEF * gain / WIND_ALIGN_SCALE


def _wind_scan_term(
    *,
    dz: float,
    curr_horizontal: float,
    max_probe: float,
    wind_toward: float,
) -> float:
    if dz < WIND_SCAN_MIN_DZ_M or curr_horizontal > WIND_SCAN_MAX_DIST_M:
        return 0.0
    gain = max(0.0, max_probe - wind_toward)
    if gain <= 0.0:
        return 0.0
    return WIND_SCAN_DELTA_COEF * gain / WIND_ALIGN_SCALE


def compute_reward(
    wind_interp: WindInterpolator,
    step: SimResult,
    ctx: RewardStepContext,
    state: RewardState,
) -> RewardResult:
    del wind_interp

    target = np.asarray(step.target_position, dtype=np.float32)
    current_position = np.asarray(step.position, dtype=np.float32)
    previous_position = np.asarray(ctx.previous_position, dtype=np.float32)
    wind = np.asarray(step.wind, dtype=np.float32)

    prev_horizontal = _horizontal_distance(target, previous_position)
    curr_horizontal = _horizontal_distance(target, current_position)
    horizontal_progress = prev_horizontal - curr_horizontal

    prev_vertical = _vertical_distance(target, previous_position)
    curr_vertical = _vertical_distance(target, current_position)
    vertical_progress = prev_vertical - curr_vertical

    prev_distance_3d = _distance_3d(target, previous_position)
    curr_distance_3d = _distance_3d(target, current_position)

    wind_toward = _wind_toward(target, current_position, wind)
    wind_align_delta = wind_toward - state.prev_wind_toward
    if wind_toward < 0.0:
        state.adverse_wind_steps += 1
    else:
        state.adverse_wind_steps = 0
    state.last_wind_align_delta = wind_align_delta

    if horizontal_progress < 0.0:
        state.consecutive_negative_horizontal_progress += 1
    else:
        state.consecutive_negative_horizontal_progress = 0

    if curr_horizontal < state.best_horizontal_distance:
        state.best_horizontal_distance = curr_horizontal

    max_probe = (
        float(ctx.max_probe_wind_toward)
        if ctx.max_probe_wind_toward is not None
        else wind_toward
    )

    pbrs_term = _pbrs_term(prev_distance_3d, curr_distance_3d)
    progress_term = 0.0

    wind_align_term = WIND_ALIGN_COEF * wind_toward / WIND_ALIGN_SCALE
    wind_align_delta_term = WIND_ALIGN_DELTA_COEF * wind_align_delta / WIND_ALIGN_SCALE

    dz = abs(float(current_position[2] - previous_position[2]))
    climbing = float(current_position[2]) > float(previous_position[2]) + 1e-3

    probe_layer_term = _probe_layer_term(
        climbing=climbing and dz >= PROBE_LAYER_MIN_DZ_M,
        max_probe=max_probe,
        wind_toward=wind_toward,
    )
    wind_scan_term = _wind_scan_term(
        dz=dz,
        curr_horizontal=curr_horizontal,
        max_probe=max_probe,
        wind_toward=wind_toward,
    )

    energy_term = _energy_term(ctx.energy_delta, dz)
    boundary_term = -BOUNDARY_PENALTY if ctx.boundary_contact else 0.0

    state.consecutive_favorable_wind = 0
    state.consecutive_adverse_wind = 0
    state.idle_action_streak = 0

    reward = (
        pbrs_term
        + progress_term
        + wind_align_term
        + wind_align_delta_term
        + probe_layer_term
        + wind_scan_term
        + energy_term
        + boundary_term
    )

    terminated = bool(
        curr_horizontal <= ctx.target_reach_radius
        and curr_vertical <= ctx.target_vertical_reach_radius
    )
    truncated = bool(ctx.step_count >= ctx.max_episode_steps)
    if terminated:
        reward += SUCCESS_REWARD

    state.prev_wind_toward = wind_toward

    terms = {
        "reward_pbrs_term": pbrs_term,
        "reward_progress_term": progress_term,
        "reward_wind_align_term": wind_align_term,
        "reward_wind_align_delta_term": wind_align_delta_term,
        "reward_probe_layer_term": probe_layer_term,
        "reward_wind_scan_term": wind_scan_term,
        "reward_energy_term": energy_term,
        "reward_boundary_term": boundary_term,
        "reward_distance_term": 0.0,
        "reward_best_distance_term": 0.0,
        "reward_distance_regression_term": 0.0,
        "reward_hold_close_term": 0.0,
        "reward_wind_streak_term": 0.0,
        "reward_wind_adverse_streak_term": 0.0,
        "reward_adverse_wind_close_term": 0.0,
        "reward_high_altitude_term": 0.0,
        "reward_idle_action_term": 0.0,
        "reward_z_stick_term": 0.0,
    }

    return RewardResult(
        reward=float(reward),
        terminated=terminated,
        truncated=truncated,
        horizontal_progress=horizontal_progress,
        vertical_progress=vertical_progress,
        wind_toward=wind_toward,
        wind_align_delta=wind_align_delta,
        terms=terms,
        consecutive_favorable_wind=state.consecutive_favorable_wind,
        consecutive_adverse_wind=state.consecutive_adverse_wind,
    )
