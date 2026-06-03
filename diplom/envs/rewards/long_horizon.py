# Reward long_horizon: shaping под длинный горизонт.

from __future__ import annotations

import numpy as np

from diplom.envs.rewards.types import RewardResult, RewardState, RewardStepContext
from diplom.sim.simulation import SimResult
from diplom.wind.interp import WindInterpolator

HORIZONTAL_PROGRESS_SCALE = 200.0
VERTICAL_PROGRESS_SCALE = 4000.0
PROGRESS_ZONE_FAR_M = 20_000.0
PROGRESS_ZONE_MID_M = 5_000.0
PROGRESS_ZONE_NEAR_M = 1_000.0
PROGRESS_ZONE_MID_MULT = 3.0
PROGRESS_ZONE_NEAR_MULT = 10.0
PROGRESS_ZONE_FINISH_MULT = 25.0
HORIZONTAL_PROGRESS_POS_COEF = 2.0
HORIZONTAL_PROGRESS_NEG_COEF = 0.15
VERTICAL_PROGRESS_POS_COEF = 0.25
VERTICAL_PROGRESS_NEG_COEF = 0.02
WIND_ALIGN_SCALE = 20.0
WIND_ALIGN_COEF = 0.8
WIND_ALIGN_DELTA_COEF = 0.4
WIND_SCAN_MIN_DZ_M = 10.0
WIND_SCAN_DELTA_COEF = 3.0
WIND_SCAN_MAX_DIST_M = 50_000.0
FAVORABLE_WIND_FAR_RADIUS_M = 5_000.0
FAVORABLE_WIND_FAR_THRESHOLD = 0.5
FAVORABLE_WIND_FAR_BONUS = 0.03
BEST_DISTANCE_BONUS = 30.0
BEST_DISTANCE_MAX_DIST_M = 50_000.0
HOLD_CLOSE_RADIUS_M = 5_000.0
HOLD_CLOSE_BONUS = 0.05
ENERGY_COEF = 0.3
ENERGY_SCALE = 100.0
BOUNDARY_PENALTY = 0.05
PBRS_GAMMA = 0.99
PBRS_COEF = 0.5
PBRS_DISTANCE_SCALE = 75_000.0
SUCCESS_REWARD = 500.0
Z_STICK_WINDOW_STEPS = 1


def _horizontal_distance(target: np.ndarray, position: np.ndarray) -> float:
    return float(np.linalg.norm((target[:2] - position[:2]).astype(np.float64)))


def _vertical_distance(target: np.ndarray, position: np.ndarray) -> float:
    return float(abs(float(target[2]) - float(position[2])))


def _wind_toward(target: np.ndarray, position: np.ndarray, wind: np.ndarray) -> float:
    delta_xy = target[:2] - position[:2]
    norm = float(np.linalg.norm(delta_xy))
    if norm < 1e-6:
        return 0.0
    unit = delta_xy / norm
    return float(wind[0] * unit[0] + wind[1] * unit[1])


def _horizontal_progress_zone_multiplier(dist_xy: float) -> float:
    if dist_xy <= PROGRESS_ZONE_NEAR_M:
        return PROGRESS_ZONE_FINISH_MULT
    if dist_xy <= PROGRESS_ZONE_MID_M:
        return PROGRESS_ZONE_NEAR_MULT
    if dist_xy <= PROGRESS_ZONE_FAR_M:
        return PROGRESS_ZONE_MID_MULT
    return 1.0


def _asymmetric_progress_term(
    progress: float,
    scale: float,
    pos_coef: float,
    neg_coef: float,
) -> float:
    if progress >= 0.0:
        return pos_coef * progress / scale
    return neg_coef * progress / scale


def _pbrs_term(prev_horizontal: float, curr_horizontal: float) -> float:
    # PBRS: gamma * Phi(s') - Phi(s), Phi(s) = -coef * dist / scale.
    delta = prev_horizontal - PBRS_GAMMA * curr_horizontal
    return PBRS_COEF * delta / PBRS_DISTANCE_SCALE


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

    wind_align_term = WIND_ALIGN_COEF * wind_toward / WIND_ALIGN_SCALE
    wind_align_delta_term = WIND_ALIGN_DELTA_COEF * wind_align_delta / WIND_ALIGN_SCALE

    progress_zone_mult = _horizontal_progress_zone_multiplier(curr_horizontal)
    horizontal_progress_term = progress_zone_mult * _asymmetric_progress_term(
        horizontal_progress,
        HORIZONTAL_PROGRESS_SCALE,
        HORIZONTAL_PROGRESS_POS_COEF,
        HORIZONTAL_PROGRESS_NEG_COEF,
    )
    progress_term = horizontal_progress_term + _asymmetric_progress_term(
        vertical_progress,
        VERTICAL_PROGRESS_SCALE,
        VERTICAL_PROGRESS_POS_COEF,
        VERTICAL_PROGRESS_NEG_COEF,
    )

    pbrs_term = _pbrs_term(prev_horizontal, curr_horizontal)

    energy_term = -ENERGY_COEF * ctx.energy_delta / ENERGY_SCALE
    boundary_term = -BOUNDARY_PENALTY if ctx.boundary_contact else 0.0

    best_distance_term = 0.0
    if curr_horizontal < state.best_horizontal_distance:
        if curr_horizontal <= BEST_DISTANCE_MAX_DIST_M:
            best_distance_term = BEST_DISTANCE_BONUS * (prev_horizontal / BEST_DISTANCE_MAX_DIST_M)
        state.best_horizontal_distance = curr_horizontal

    hold_close_term = HOLD_CLOSE_BONUS if curr_horizontal < HOLD_CLOSE_RADIUS_M else 0.0

    dz = abs(float(current_position[2] - previous_position[2]))
    wind_scan_term = 0.0
    if dz >= WIND_SCAN_MIN_DZ_M and wind_align_delta > 0.0 and curr_horizontal <= WIND_SCAN_MAX_DIST_M:
        wind_scan_term = WIND_SCAN_DELTA_COEF * wind_align_delta / WIND_ALIGN_SCALE

    favorable_wind_far_term = 0.0
    if curr_horizontal >= FAVORABLE_WIND_FAR_RADIUS_M and wind_toward >= FAVORABLE_WIND_FAR_THRESHOLD:
        favorable_wind_far_term = FAVORABLE_WIND_FAR_BONUS

    state.consecutive_favorable_wind = 0
    state.consecutive_adverse_wind = 0
    state.idle_action_streak = 0

    reward = (
        wind_align_term
        + wind_align_delta_term
        + progress_term
        + pbrs_term
        + energy_term
        + boundary_term
        + best_distance_term
        + hold_close_term
        + wind_scan_term
        + favorable_wind_far_term
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
        "reward_wind_align_term": wind_align_term,
        "reward_wind_align_delta_term": wind_align_delta_term,
        "reward_progress_term": progress_term,
        "reward_pbrs_term": pbrs_term,
        "reward_energy_term": energy_term,
        "reward_boundary_term": boundary_term,
        "reward_best_distance_term": best_distance_term,
        "reward_hold_close_term": hold_close_term,
        "reward_wind_scan_term": wind_scan_term,
        "reward_favorable_wind_far_term": favorable_wind_far_term,
        "reward_distance_term": 0.0,
        "reward_distance_regression_term": 0.0,
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
