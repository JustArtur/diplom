from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from diplom.sim.simulation import SimParams, Simulation, SimResult
from diplom.wind.interp import WindInterpolator
from .constants import ACTION_LIMIT, DEFAULT_DT
from ..sim.constants import WORLD_SIZE, MIN_HEIGHT, DEFAULT_AIR_WEIGHT


@dataclass
class EnvConfig:
    wind_interp: WindInterpolator
    initial_position: np.ndarray = np.array([
            np.random.uniform(-WORLD_SIZE, WORLD_SIZE),
            np.random.uniform(-WORLD_SIZE, WORLD_SIZE),
            np.random.uniform(MIN_HEIGHT, np.inf)
        ], dtype=np.float64)
    initial_target_position: np.ndarray = np.array([
            np.random.uniform(-WORLD_SIZE, WORLD_SIZE),
            np.random.uniform(-WORLD_SIZE, WORLD_SIZE),
            np.random.uniform(MIN_HEIGHT, np.inf)
        ], dtype=np.float64)
    dt: float = DEFAULT_DT


class BalloonEnv(gym.Env):
    """Gymnasium-среда управления стратостатом."""

    metadata = {"render_modes": ["ansi"], "render_fps": 20}

    def __init__(self, config: EnvConfig) -> None:
        self.wind_interp: WindInterpolator = config.wind_interp
        self.sim: Simulation = Simulation(SimParams(wind_interp=self.wind_interp, target_position=config.initial_target_position, initial_position=config.initial_position))

        self.initial_position: np.ndarray = config.initial_position
        self.initial_target_position: np.ndarray = config.initial_target_position
        self.dt: float = config.dt

        self.action_space = spaces.Box(low=-ACTION_LIMIT, high=ACTION_LIMIT, shape=(1,), dtype=np.float64)
        self.observation_space = spaces.Dict(
            {
                "position": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
                "target_position": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
                "delta_position": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
                "wind": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float64),
                "energy": spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64),
                "air_weight": spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64),
                "vertical_speed": spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64),
                "vertical_acceleration": spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64),
                "air_density": spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64),
                "temperature": spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64),
                "pressure": spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64),
            }
        )

    # ---- gym API ----
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)

        sim_result = self.sim.step(0, 0)
        info = {}
        return self.to_obs(sim_result=sim_result), info

    def step(self, action):
        clipped_action = float(np.clip(np.asarray(action, dtype=np.float64), -ACTION_LIMIT, ACTION_LIMIT)[0])
        prev_position = np.array(self.sim.position, dtype=np.float64)
        prev_energy = float(self.sim.energy_spent)

        result = self.sim.step(self.dt, clipped_action)

        prev_distance = float(np.linalg.norm(result.target_position - prev_position))
        current_distance = float(np.linalg.norm(result.target_position - result.position))
        progress_reward = prev_distance - current_distance

        energy_delta = float(result.energy_spent - prev_energy)
        energy_penalty = 0.01 * energy_delta

        reward = progress_reward - energy_penalty

        truncated = False
        terminated = False
        if current_distance <= 25.0:
            terminated = True
            reward += 100

        info = {
            "action": clipped_action,
            "progress_reward": progress_reward,
            "energy_penalty": energy_penalty,
            "distance_to_target": current_distance,
        }

        obs = self.to_obs(sim_result=result)

        return obs, float(reward), terminated, truncated, info

    def to_obs(self, sim_result: SimResult) -> dict[str, np.ndarray]:
        return {
            "position": sim_result.position,
            "target_position": sim_result.target_position,
            "delta_position": sim_result.target_position - sim_result.position,
            "wind": sim_result.wind,
            "energy": sim_result.energy_spent,
            "air_weight": sim_result.air_weight,
            "vertical_speed": sim_result.vertical_speed,
            "vertical_acceleration": sim_result.vertical_acceleration,
            "air_density": sim_result.air_density,
            "temperature": sim_result.temperature,
            "pressure": sim_result.pressure,
        }
