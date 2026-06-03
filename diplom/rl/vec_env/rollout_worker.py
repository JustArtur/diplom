from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def build_worker_policy(
    env: gym.Env,
    *,
    learning_rate: float = 3e-4,
    model_name: str = "default",
) -> Any:
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.utils import ConstantSchedule

    from diplom.rl.ppo.policy import build_ppo_policy_kwargs

    return ActorCriticPolicy(
        env.observation_space,
        env.action_space,
        ConstantSchedule(learning_rate),
        **build_ppo_policy_kwargs(model_name),
    )


def prepare_env_action(policy: Any, action_space: spaces.Space, action: np.ndarray) -> np.ndarray:
    clipped = np.asarray(action, dtype=np.float32).reshape(-1)
    if isinstance(action_space, spaces.Box):
        if policy.squash_output:
            return policy.unscale_action(clipped)
        return np.clip(clipped, action_space.low, action_space.high)
    return clipped
