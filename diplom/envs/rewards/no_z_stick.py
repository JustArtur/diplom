# Reward no_z_stick: без z_stick.

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
HORIZONTAL_DISTANCE_SCALE = 75_000.0
HORIZONTAL_DISTANCE_COEF = 0.01
DISTANCE_NEAR_RADIUS_M = 50_000.0
DISTANCE_NEAR_QUAD_COEF = 0.5
WIND_ALIGN_SCALE = 20.0
WIND_ALIGN_COEF = 0.5
WIND_ALIGN_DELTA_COEF = 0.25
WIND_ALIGN_ADVERSE_PROGRESS_SCALE = 0.25
WIND_ALIGN_ZERO_PROGRESS_STEPS = 5
WIND_FAVORABLE_THRESHOLD = 0.0
WIND_ADVERSE_THRESHOLD = -5.0
WIND_FAVORABLE_STREAK_STEPS = 80
WIND_ADVERSE_STREAK_STEPS = 200
WIND_FAVORABLE_STREAK_BONUS = 0.04
WIND_ADVERSE_STREAK_PENALTY = 0.1
HORIZONTAL_PROGRESS_POS_COEF = 3.0
HORIZONTAL_PROGRESS_NEG_COEF = 0.4
VERTICAL_PROGRESS_POS_COEF = 0.25
VERTICAL_PROGRESS_NEG_COEF = 0.05
BEST_DISTANCE_BONUS = 20.0
BEST_DISTANCE_MAX_DIST_M = 50_000.0
DISTANCE_REGRESSION_COEF = 0.5
DISTANCE_REGRESSION_SCALE_M = 1_000.0
HOLD_CLOSE_RADIUS_M = 5_000.0
HOLD_CLOSE_BONUS = 0.05
ENERGY_COEF = 0.5
ENERGY_SCALE = 100.0
BOUNDARY_PENALTY = 0.05
HIGH_ALTITUDE_M = 5000.0
HIGH_ALTITUDE_ADVERSE_PENALTY = 0.05
IDLE_ACTION_THRESHOLD = 0.3
IDLE_ACTION_MIN_DZ_M = 0.1
IDLE_ACTION_STREAK_STEPS = 40
IDLE_ACTION_PENALTY = 0.02
WIND_SCAN_MIN_DZ_M = 20.0
WIND_SCAN_DELTA_COEF = 2.0
WIND_SCAN_MAX_DIST_M = 30_000.0
ADVERSE_WIND_CLOSE_RADIUS_M = 10_000.0
ADVERSE_WIND_CLOSE_PENALTY = 0.15
Z_STICK_WINDOW_STEPS = 1
Z_STICK_MIN_STD_M = 200.0
Z_STICK_PENALTY = 0.03
SUCCESS_REWARD = 500.0


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


def _wind_streak_terms(
    state: RewardState,
    wind_toward: float,
    horizontal_progress: float,
) -> tuple[float, float]:
    if wind_toward > WIND_FAVORABLE_THRESHOLD:
        state.consecutive_favorable_wind += 1
        state.consecutive_adverse_wind = 0
    elif wind_toward < WIND_ADVERSE_THRESHOLD:
        state.consecutive_adverse_wind += 1
        state.consecutive_favorable_wind = 0
    else:
        state.consecutive_favorable_wind = 0
        state.consecutive_adverse_wind = 0

    streak_term = 0.0
    if state.consecutive_favorable_wind >= WIND_FAVORABLE_STREAK_STEPS and horizontal_progress > 0.0:
        streak_term = WIND_FAVORABLE_STREAK_BONUS

    adverse_streak_term = 0.0
    if state.consecutive_adverse_wind >= WIND_ADVERSE_STREAK_STEPS:
        adverse_streak_term = -WIND_ADVERSE_STREAK_PENALTY

    return streak_term, adverse_streak_term


def compute_reward(
    wind_interp: WindInterpolator,
    step: SimResult,
    ctx: RewardStepContext,
    state: RewardState,
) -> RewardResult:
    del wind_interp  # зарезервировано для reward-термов с пробами ветра по высоте

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

    if state.consecutive_negative_horizontal_progress >= WIND_ALIGN_ZERO_PROGRESS_STEPS:
        wind_progress_scale = 0.0
    elif horizontal_progress >= 0.0:
        wind_progress_scale = 1.0
    else:
        wind_progress_scale = WIND_ALIGN_ADVERSE_PROGRESS_SCALE

    wind_align_term = wind_progress_scale * WIND_ALIGN_COEF * wind_toward / WIND_ALIGN_SCALE
    wind_align_delta_term = (
        wind_progress_scale * WIND_ALIGN_DELTA_COEF * wind_align_delta / WIND_ALIGN_SCALE
    )
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
    distance_term = -HORIZONTAL_DISTANCE_COEF * curr_horizontal / HORIZONTAL_DISTANCE_SCALE
    if curr_horizontal < DISTANCE_NEAR_RADIUS_M:
        ratio = curr_horizontal / HORIZONTAL_DISTANCE_SCALE
        distance_term -= DISTANCE_NEAR_QUAD_COEF * ratio * ratio

    energy_term = -ENERGY_COEF * ctx.energy_delta / ENERGY_SCALE
    boundary_term = -BOUNDARY_PENALTY if ctx.boundary_contact else 0.0

    best_distance_term = 0.0
    if curr_horizontal < state.best_horizontal_distance:
        if curr_horizontal <= BEST_DISTANCE_MAX_DIST_M:
            best_distance_term = BEST_DISTANCE_BONUS * (prev_horizontal / BEST_DISTANCE_MAX_DIST_M)
        state.best_horizontal_distance = curr_horizontal

    regression_m = max(0.0, curr_horizontal - state.best_horizontal_distance)
    regression_term = -DISTANCE_REGRESSION_COEF * regression_m / DISTANCE_REGRESSION_SCALE_M

    hold_close_term = HOLD_CLOSE_BONUS if curr_horizontal < HOLD_CLOSE_RADIUS_M else 0.0

    wind_streak_term, wind_adverse_streak_term = _wind_streak_terms(
        state,
        wind_toward,
        horizontal_progress,
    )

    dz = abs(float(current_position[2] - previous_position[2]))
    wind_scan_term = 0.0
    if dz >= WIND_SCAN_MIN_DZ_M and wind_align_delta > 0.0 and curr_horizontal <= WIND_SCAN_MAX_DIST_M:
        wind_scan_term = WIND_SCAN_DELTA_COEF * wind_align_delta / WIND_ALIGN_SCALE

    adverse_wind_close_term = 0.0
    if curr_horizontal < ADVERSE_WIND_CLOSE_RADIUS_M and wind_toward < 0.0:
        adverse_wind_close_term = -ADVERSE_WIND_CLOSE_PENALTY

    high_altitude_term = 0.0
    if current_position[2] > HIGH_ALTITUDE_M and wind_toward < 0.0:
        high_altitude_term = -HIGH_ALTITUDE_ADVERSE_PENALTY

    if abs(ctx.clipped_action) >= IDLE_ACTION_THRESHOLD and dz < IDLE_ACTION_MIN_DZ_M:
        state.idle_action_streak += 1
    else:
        state.idle_action_streak = 0

    idle_action_term = 0.0
    if state.idle_action_streak >= IDLE_ACTION_STREAK_STEPS:
        idle_action_term = -IDLE_ACTION_PENALTY

    z_stick_term = 0.0

    reward = (
        wind_align_term
        + wind_align_delta_term
        + progress_term
        + distance_term
        + energy_term
        + boundary_term
        + best_distance_term
        + regression_term
        + hold_close_term
        + wind_streak_term
        + wind_adverse_streak_term
        + wind_scan_term
        + adverse_wind_close_term
        + high_altitude_term
        + idle_action_term
        + z_stick_term
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
        "reward_distance_term": distance_term,
        "reward_energy_term": energy_term,
        "reward_boundary_term": boundary_term,
        "reward_best_distance_term": best_distance_term,
        "reward_distance_regression_term": regression_term,
        "reward_hold_close_term": hold_close_term,
        "reward_wind_streak_term": wind_streak_term,
        "reward_wind_adverse_streak_term": wind_adverse_streak_term,
        "reward_wind_scan_term": wind_scan_term,
        "reward_adverse_wind_close_term": adverse_wind_close_term,
        "reward_high_altitude_term": high_altitude_term,
        "reward_idle_action_term": idle_action_term,
        "reward_z_stick_term": z_stick_term,
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
