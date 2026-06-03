from __future__ import annotations

import numpy as np

from diplom.envs.rewards.types import RewardResult, RewardState, RewardStepContext
from diplom.sim.simulation import SimResult
from diplom.wind.interp import WindInterpolator

NEEDS_PROBE_WINDS = False
WIND_ALIGN_SCALE = 20.0
Z_STICK_WINDOW_STEPS = 1
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

    if wind_toward > 0.0:
        state.consecutive_favorable_wind += 1
        state.consecutive_adverse_wind = 0
    elif wind_toward < -5.0:
        state.consecutive_adverse_wind += 1
        state.consecutive_favorable_wind = 0
    else:
        state.consecutive_favorable_wind = 0
        state.consecutive_adverse_wind = 0

    if wind_toward < 0.0:
        state.adverse_wind_steps += 1
    else:
        state.adverse_wind_steps = 0

    state.last_wind_align_delta = wind_align_delta
    if curr_horizontal < state.best_horizontal_distance:
        state.best_horizontal_distance = curr_horizontal

    state.consecutive_negative_horizontal_progress = 0
    state.idle_action_streak = 0

    terminated = bool(
        curr_horizontal <= ctx.target_reach_radius
        and curr_vertical <= ctx.target_vertical_reach_radius
    )
    truncated = bool(ctx.step_count >= ctx.max_episode_steps)

    goal_term = SUCCESS_REWARD if terminated else 0.0
    reward = goal_term

    state.prev_wind_toward = wind_toward

    terms = {
        "reward_goal_term": goal_term,
        "reward_wind_align_term": 0.0,
        "reward_wind_align_delta_term": 0.0,
        "reward_progress_term": 0.0,
        "reward_distance_term": 0.0,
        "reward_energy_term": 0.0,
        "reward_boundary_term": 0.0,
        "reward_best_distance_term": 0.0,
        "reward_distance_regression_term": 0.0,
        "reward_hold_close_term": 0.0,
        "reward_wind_streak_term": 0.0,
        "reward_wind_adverse_streak_term": 0.0,
        "reward_wind_scan_term": 0.0,
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
