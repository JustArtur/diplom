from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional

import numpy as np
from stable_baselines3 import PPO

from diplom.config import AppConfig
from diplom.envs.factory import build_env
from diplom.torch_device import resolve_torch_device
from diplom.world import log_world_bounds


@dataclass
class EpisodeResult:
    success: bool
    total_reward: float
    steps: int
    trajectory: List[dict[str, object]]
    target_position: List[float] = field(default_factory=list)


def rollout_episodes(
    config: AppConfig,
    n_episodes: int = 1,
    policy_path: Optional[str] = None,
    render: bool = False,
    seed: int = 0,
) -> List[EpisodeResult]:
    env_config = replace(
        config.environment,
        randomize_initial_position=False,
        randomize_target_position=False,
    )
    env = build_env(env_config, config.wind, env_idx=0)
    log_world_bounds(
        env.world_bounds,
        origin_lat=env.wind_interp.origin_lat,
        origin_lon=env.wind_interp.origin_lon,
        wind_path=config.wind.path,
        prefix="[rollout]",
    )
    device = resolve_torch_device(config.training.device)
    model = (
        PPO.load(policy_path, device=device)
        if policy_path is not None
        else None
    )
    results: List[EpisodeResult] = []

    try:
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + ep)
            done = False
            truncated = False
            total_reward = 0.0
            traj: list[dict[str, object]] = []

            while not (done or truncated):
                if model is not None:
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = env.action_space.sample()

                obs, reward, done, truncated, _info = env.step(action)
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
                    }
                )

                if render:
                    print(env.render())  # noqa: T201 - CLI вывод

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
        env.close()

    return results
