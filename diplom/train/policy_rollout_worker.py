"""Общие хелперы для policy rollout в subprocess."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def build_worker_policy(env: gym.Env) -> Any:
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.utils import ConstantSchedule

    return ActorCriticPolicy(
        env.observation_space,
        env.action_space,
        ConstantSchedule(3e-4),
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
    )


def prepare_env_action(policy: Any, action_space: spaces.Space, action: np.ndarray) -> np.ndarray:
    clipped = np.asarray(action, dtype=np.float32).reshape(-1)
    if isinstance(action_space, spaces.Box):
        if policy.squash_output:
            return policy.unscale_action(clipped)
        return np.clip(clipped, action_space.low, action_space.high)
    return clipped
