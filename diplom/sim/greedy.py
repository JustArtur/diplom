from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diplom.envs.constants import TARGET_VERTICAL_REACH_RADIUS
from diplom.sim.simulation import Simulation


@dataclass(frozen=True, slots=True)
class GreedyActionChoice:
    action: float
    score: float
    lookahead_steps: int
    candidate_count: int
    final_horizontal_distance: float
    final_vertical_distance: float
    success: bool


def _candidate_actions(action_limit: float, candidate_count: int) -> np.ndarray:
    count = max(3, int(candidate_count))
    grid = np.linspace(-action_limit, action_limit, num=count, dtype=np.float32)
    if not np.any(np.isclose(grid, 0.0)):
        grid = np.concatenate([grid, np.array([0.0], dtype=np.float32)])
    return np.unique(grid)


def _distance_metrics(sim: Simulation) -> tuple[float, float]:
    target = np.asarray(sim.target_position, dtype=np.float64)
    position = np.asarray(sim.position, dtype=np.float64)
    horizontal_distance = float(np.linalg.norm(target[:2] - position[:2]))
    vertical_distance = float(abs(target[2] - position[2]))
    return horizontal_distance, vertical_distance


def _evaluate_action(
    sim: Simulation,
    action: float,
    *,
    dt: float,
    lookahead_steps: int,
    target_reach_radius: float,
    vertical_weight: float,
) -> GreedyActionChoice:
    clone = sim.clone()
    final_horizontal_distance = float("inf")
    final_vertical_distance = float("inf")

    for step_idx in range(max(1, int(lookahead_steps))):
        clone.step(dt, action)
        final_horizontal_distance, final_vertical_distance = _distance_metrics(clone)
        if (
            final_horizontal_distance <= target_reach_radius
            and final_vertical_distance <= TARGET_VERTICAL_REACH_RADIUS
        ):
            return GreedyActionChoice(
                action=float(action),
                score=-1e12 + float(step_idx),
                lookahead_steps=max(1, int(lookahead_steps)),
                candidate_count=0,
                final_horizontal_distance=final_horizontal_distance,
                final_vertical_distance=final_vertical_distance,
                success=True,
            )

    score = final_horizontal_distance + vertical_weight * final_vertical_distance
    return GreedyActionChoice(
        action=float(action),
        score=float(score),
        lookahead_steps=max(1, int(lookahead_steps)),
        candidate_count=0,
        final_horizontal_distance=final_horizontal_distance,
        final_vertical_distance=final_vertical_distance,
        success=False,
    )


def choose_greedy_action(
    sim: Simulation,
    *,
    dt: float,
    action_limit: float,
    target_reach_radius: float,
    lookahead_steps: int = 6,
    candidate_count: int = 9,
    vertical_weight: float = 20.0,
) -> GreedyActionChoice:
    actions = _candidate_actions(action_limit, candidate_count)
    evaluations = [
        _evaluate_action(
            sim,
            float(action),
            dt=dt,
            lookahead_steps=lookahead_steps,
            target_reach_radius=target_reach_radius,
            vertical_weight=vertical_weight,
        )
        for action in actions
    ]
    best = min(evaluations, key=lambda item: (item.score, abs(item.action)))
    return GreedyActionChoice(
        action=best.action,
        score=best.score,
        lookahead_steps=best.lookahead_steps,
        candidate_count=len(actions),
        final_horizontal_distance=best.final_horizontal_distance,
        final_vertical_distance=best.final_vertical_distance,
        success=best.success,
    )
