from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from diplom.sim.simulation import SimParams, Simulation
from diplom.wind.interp import WindInterpolator


@dataclass
class EnvConfig:
    params: SimParams
    max_steps: int = 1_200
    start_time: Optional[datetime] = None
    wind_scale: float = 30.0  # для нормализации наблюдений по ветру


class BalloonEnv(gym.Env):
    """Gymnasium-среда управления стратостатом."""

    metadata = {"render_modes": ["ansi"], "render_fps": 20}

    def __init__(self, wind_interp: WindInterpolator) -> None:
        self.wind_interp = wind_interp
        self.action_space = spaces.Box(low=-self._control_limit, high=self._control_limit, shape=(1,), dtype=np.float64)
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
            }
        )

        self._state: Simulation | None = None
        self._step_count = 0

    # ---- gym API ----
    def reset(self,) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset()

        self._state = Simulation(self.params)
        self._step_count = 0

        return self.get_obs(), {}

    def step(self, action):
        assert self._state is not None, "call reset() first"

        action_vz = float(np.asarray(action, dtype=np.float64).squeeze())
        action_vz = float(np.clip(action_vz, -self._control_limit, self._control_limit))

        prev_position = np.asarray(self._state.position, dtype=np.float64).copy()
        result = self._state.step(dt=self._dt, air_pump_speed=action_vz)
        self._step_count += 1

        obs = self.get_obs()
        distance_before = float(np.linalg.norm(prev_position - np.asarray(result.target_position, dtype=np.float64)))
        distance_after = float(np.linalg.norm(result.position - np.asarray(result.target_position, dtype=np.float64)))
        reward = (distance_before - distance_after) - 0.01 * abs(action_vz)

        success_radius = float(getattr(self.params, "success_radius_m", 100.0))
        terminated = distance_after <= success_radius
        if terminated:
            reward += 100.0

        out_of_bounds = self._is_out_of_bounds(result.position)
        out_of_energy = self._is_out_of_energy(result)
        truncated = self._step_count >= self.max_steps or out_of_bounds or out_of_energy

        info: dict[str, Any] = {
            "step": self._step_count,
            "success": terminated,
            "out_of_bounds": out_of_bounds,
            "out_of_energy": out_of_energy,
            "state": {
                "position": np.asarray(result.position, dtype=np.float64),
                "target_position": np.asarray(result.target_position, dtype=np.float64),
                "delta_position": np.asarray(result.target_position - result.position, dtype=np.float64),
                "wind": np.asarray(result.wind, dtype=np.float64),
                "vertical_speed": float(result.vertical_speed),
                "vertical_acceleration": float(result.vertical_acceleration),
                "energy": float(result.energy_spent),
                "air_weight": float(result.air_weight),
                "air_density": float(result.air_density),
                "temperature": float(result.temperature),
                "pressure": float(result.pressure),
            },
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "ansi":
            return None
        assert self._state is not None
        position = np.asarray(self._state.position, dtype=np.float64)
        target = np.asarray(self._state.target_position, dtype=np.float64)
        dist = float(np.linalg.norm(target - position))
        return (
            f"pos=({position[0]:.1f},{position[1]:.1f},{position[2]:.1f}) "
            f"v_z={self._state.vertical_speed:.2f} energy={self._state.energy_spent:.1f} "
            f"dist_goal={dist:.1f}"
        )

    def get_obs(self) -> dict[str, np.ndarray]:
        assert self._state is not None, "call reset() first"

        position = np.asarray(self._state.position, dtype=np.float64)
        target_position = np.asarray(self._state.target_position, dtype=np.float64)
        delta_position = target_position - position
        wind_scale = float(self.config.wind_scale) if self.config.wind_scale else 1.0
        if wind_scale <= 0.0:
            wind_scale = 1.0

        return {
            "position": position,
            "target_position": target_position,
            "delta_position": delta_position,
            "wind": np.asarray(self._state.wind, dtype=np.float64) / wind_scale,
            "energy": np.asarray(self._state.energy_spent, dtype=np.float64),
            "air_weight": np.asarray(self._state.air_weight, dtype=np.float64),
            "vertical_speed": np.asarray(self._state.vertical_speed, dtype=np.float64),
            "vertical_acceleration": np.asarray(self._state.vertical_acceleration, dtype=np.float64),
            "air_density": np.asarray(self._state.air_density, dtype=np.float64),
            "temperature": np.asarray(self._state.temperature, dtype=np.float64),
            "pressure": np.asarray(self._state.pressure, dtype=np.float64),
        }

    @property
    def _dt(self) -> float:
        dt = getattr(self.params, "dt", 1.0)
        return float(dt) if float(dt) > 0.0 else 1.0

    def _is_out_of_bounds(self, position: np.ndarray) -> bool:
        boundary_radius = getattr(self.params, "boundary_radius_m", None)
        if boundary_radius is not None and float(boundary_radius) > 0.0:
            horizontal_distance = float(np.linalg.norm(position[:2]))
            if horizontal_distance > float(boundary_radius):
                return True

        min_z = getattr(self.params, "min_z", None)
        if min_z is not None and position[2] < float(min_z):
            return True

        max_z = getattr(self.params, "max_z", None)
        if max_z is not None and position[2] > float(max_z):
            return True

        return False

    def _is_out_of_energy(self, result: Any) -> bool:
        energy_budget = getattr(self.params, "energy_budget", None)
        if energy_budget is not None:
            return float(result.energy_spent) >= float(energy_budget)
        return False
