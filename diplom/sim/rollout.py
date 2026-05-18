from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional

import numpy as np
from stable_baselines3 import PPO

from diplom.config import AppConfig
from diplom.envs.factory import build_env
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
    env_config = replace(config.environment, randomize_start_state=False)
    env = build_env(env_config, config.wind, env_idx=0)
    log_world_bounds(
        env.world_bounds,
        origin_lat=env.wind_interp.origin_lat,
        origin_lon=env.wind_interp.origin_lon,
        wind_path=config.wind.path,
        prefix="[rollout]",
    )
    model = PPO.load(policy_path) if policy_path is not None else None
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

                obs, reward, done, truncated, info = env.step(action)
                total_reward += float(reward)
                traj.append(
                    {
                        "action": float(info["action"]),
                        "distance_to_target": float(info["distance_to_target"]),
                        "reward": float(reward),
                        "terminated": bool(done),
                        "truncated": bool(truncated),
                        "position": np.array(info["position"], dtype=np.float32).tolist(),
                        "wind": np.array(info["wind"], dtype=np.float32).tolist(),
                        "sim_time": str(info["sim_time"]),
                        "target_position": np.array(info["target_position"], dtype=np.float32).tolist(),
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
