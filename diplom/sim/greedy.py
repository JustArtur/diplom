from __future__ import annotations

from dataclasses import dataclass, replace
from multiprocessing import get_context
from pathlib import Path
from typing import Any, List

import numpy as np

from diplom.config import AppConfig
from diplom.envs.constants import TARGET_VERTICAL_REACH_RADIUS
from diplom.envs.factory import build_env
from diplom.sim.rollout import EpisodeResult
from diplom.sim.simulation import Simulation
from diplom.world import WorldBounds, log_world_bounds


@dataclass(frozen=True, slots=True)
class GreedyActionChoice:
    action: float
    score: float
    lookahead_steps: int
    candidate_count: int
    final_horizontal_distance: float
    final_vertical_distance: float
    success: bool


@dataclass(frozen=True, slots=True)
class GreedyRunConfig:
    # Единственный источник дефолтов greedy baseline (CLI и choose_greedy_action).

    lookahead_steps: int = 600
    candidate_count: int = 10
    vertical_weight: float = 20.0
    trajectory_render_interval: int = 256


DEFAULT_GREEDY_RUN_CONFIG = GreedyRunConfig()


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
    greedy: GreedyRunConfig | None = None,
) -> GreedyActionChoice:
    cfg = greedy or DEFAULT_GREEDY_RUN_CONFIG
    actions = _candidate_actions(action_limit, cfg.candidate_count)
    evaluations = [
        _evaluate_action(
            sim,
            float(action),
            dt=dt,
            lookahead_steps=cfg.lookahead_steps,
            target_reach_radius=target_reach_radius,
            vertical_weight=cfg.vertical_weight,
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


def _live_trajectory_request(
    env: Any,
    *,
    step_num: int,
    world_bounds: WorldBounds,
    wind_dataset_path: Path,
    show_wind_cones: bool,
):
    from diplom.trajectory.live.render_worker import TrajectoryRenderRequest

    state = env.get_trajectory_viz_state()
    if not state:
        return None

    env_idx = int(state["env_idx"])
    current_steps_paths: dict[int, Path] = {}
    current_step_counts: dict[int, int] = {}
    step_count = int(state["current_step_count"])
    if step_count > 0:
        current_steps_paths[env_idx] = Path(state["current_steps_path"]).resolve()
        current_step_counts[env_idx] = step_count

    return TrajectoryRenderRequest(
        num_timesteps=step_num,
        n_envs=1,
        episode_counts={env_idx: int(state["episode_count"])},
        history={env_idx: list(state["history"])},
        current_steps_paths=current_steps_paths,
        current_step_counts=current_step_counts,
        world_bounds=world_bounds,
        wind_dataset_path=wind_dataset_path,
        show_wind_cones=show_wind_cones,
        combined_html=True,
    )


def _submit_live_trajectory(
    render_queue: Any,
    output_dir: Path,
    env: Any,
    *,
    step_num: int,
    world_bounds: WorldBounds,
    wind_dataset_path: Path,
    show_wind_cones: bool,
) -> None:
    from diplom.trajectory.live.render_worker import (
        render_queue_id,
        snapshot_path_for,
        submit_trajectory_render,
        write_trajectory_snapshot,
    )

    request = _live_trajectory_request(
        env,
        step_num=step_num,
        world_bounds=world_bounds,
        wind_dataset_path=wind_dataset_path,
        show_wind_cones=show_wind_cones,
    )
    if request is None or render_queue is None:
        return
    snapshot_path = snapshot_path_for(output_dir, step_num)
    write_trajectory_snapshot(snapshot_path, request)
    submit_trajectory_render(
        render_queue,
        snapshot_path,
        queue_id=render_queue_id(output_dir),
    )


def greedy_episodes(
    config: AppConfig,
    n_episodes: int = 1,
    *,
    greedy: GreedyRunConfig | None = None,
    render: bool = False,
    seed: int = 0,
    open_trajectory_viz: bool = False,
) -> List[EpisodeResult]:
    greedy_cfg = greedy or DEFAULT_GREEDY_RUN_CONFIG
    env_config = config.environment
    env = build_env(env_config, config.wind, env_idx=0)
    log_world_bounds(
        env.world_bounds,
        origin_lat=env.wind_interp.origin_lat,
        origin_lon=env.wind_interp.origin_lon,
        wind_path=config.wind.path,
        prefix="[greedy]",
    )
    results: List[EpisodeResult] = []
    traj_dir = env_config.trajectory_steps_dir
    render_interval = max(1, int(greedy_cfg.trajectory_render_interval))
    global_step = 0
    render_queue = None
    render_process = None

    if traj_dir is not None:
        from diplom.trajectory.live.callback import _open_trajectory_viewers
        from diplom.trajectory.live.render_worker import (
            cleanup_snapshots_dir,
            start_trajectory_render_worker,
            stop_trajectory_render_worker,
        )

        traj_path = Path(traj_dir)
        traj_path.mkdir(parents=True, exist_ok=True)
        render_queue, render_process = start_trajectory_render_worker(
            ctx=get_context("spawn"),
            output_dir=traj_path,
        )
        if open_trajectory_viz:
            _open_trajectory_viewers(
                traj_path,
                n_envs=1,
                world_bounds=env.world_bounds,
                combined_html=True,
            )

    try:
        for ep in range(n_episodes):
            _obs, _ = env.reset(seed=seed + ep)
            done = False
            truncated = False
            total_reward = 0.0
            traj: list[dict[str, object]] = []

            while not (done or truncated):
                choice = choose_greedy_action(
                    env.sim,
                    dt=env.dt,
                    action_limit=float(env.action_limit),
                    target_reach_radius=float(env.target_reach_radius),
                    greedy=greedy_cfg,
                )
                action = np.array([choice.action], dtype=np.float32)

                _obs, reward, done, truncated, _info = env.step(action)
                record = env.consume_step_record()
                total_reward += float(reward)
                traj.append(
                    {
                        "action": float(record["action"]),
                        "distance_to_target": float(record["distance_to_target"]),
                        "reward": float(reward),
                        "terminated": bool(done),
                        "truncated": bool(truncated),
                        "position": list(record["position"]),
                        "wind": list(record["wind"]),
                        "sim_time": str(record["sim_time"]),
                        "vertical_speed": float(record["vertical_speed"]),
                        "target_position": list(record["target_position"]),
                        "greedy_score": float(choice.score),
                        "greedy_success": bool(choice.success),
                        "greedy_lookahead_steps": int(choice.lookahead_steps),
                        "greedy_candidate_count": int(choice.candidate_count),
                    }
                )

                if render:
                    print(env.render())  # noqa: T201 - CLI вывод

                global_step += 1
                if render_queue is not None and global_step % render_interval == 0:
                    _submit_live_trajectory(
                        render_queue,
                        Path(traj_dir),
                        env,
                        step_num=global_step,
                        world_bounds=env.world_bounds,
                        wind_dataset_path=Path(config.wind.path),
                        show_wind_cones=bool(env_config.trajectory_show_wind_cones),
                    )

            if render_queue is not None and global_step > 0:
                _submit_live_trajectory(
                    render_queue,
                    Path(traj_dir),
                    env,
                    step_num=global_step,
                    world_bounds=env.world_bounds,
                    wind_dataset_path=Path(config.wind.path),
                    show_wind_cones=bool(env_config.trajectory_show_wind_cones),
                )

            target_position = traj[-1]["target_position"] if traj else []
            results.append(
                EpisodeResult(
                    success=bool(done),
                    total_reward=total_reward,
                    steps=len(traj),
                    trajectory=traj,
                    target_position=target_position,
                )
            )
    finally:
        if render_queue is not None:
            from diplom.trajectory.live.render_worker import (
                cleanup_snapshots_dir,
                stop_trajectory_render_worker,
            )

            stop_trajectory_render_worker(render_queue, render_process)
            cleanup_snapshots_dir(Path(traj_dir))
        env.close()

    return results
