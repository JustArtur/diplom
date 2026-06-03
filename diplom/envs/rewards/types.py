from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from diplom.sim.simulation import SimResult


@dataclass(frozen=True, slots=True)
class RewardStepContext:
    # Параметры одного шага среды; одинаковы для всех reward-модулей

    result: SimResult
    previous_position: np.ndarray
    clipped_action: float
    energy_delta: float
    sim_time: np.datetime64
    z_min: float
    z_max: float
    boundary_contact: bool
    step_count: int
    max_episode_steps: int
    target_reach_radius: float
    target_vertical_reach_radius: float
    max_probe_wind_toward: float | None = None


@dataclass(frozen=True, slots=True)
class RewardResult:
    reward: float
    terminated: bool
    truncated: bool
    horizontal_progress: float
    vertical_progress: float
    wind_toward: float
    wind_align_delta: float
    terms: dict[str, float]
    consecutive_favorable_wind: int
    consecutive_adverse_wind: int


@dataclass(slots=True)
class RewardState:
    prev_wind_toward: float = 0.0
    last_wind_align_delta: float = 0.0
    adverse_wind_steps: int = 0
    best_horizontal_distance: float = field(default_factory=lambda: float("inf"))
    consecutive_favorable_wind: int = 0
    consecutive_adverse_wind: int = 0
    consecutive_negative_horizontal_progress: int = 0
    idle_action_streak: int = 0
    z_window: deque[float] = field(default_factory=deque)
    z_window_sum: float = 0.0
    z_window_sumsq: float = 0.0

    def reset(
        self,
        *,
        target: np.ndarray,
        position: np.ndarray,
        wind: np.ndarray,
        wind_toward_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
        z_window_maxlen: int,
    ) -> None:
        self.prev_wind_toward = wind_toward_fn(target, position, wind)
        self.last_wind_align_delta = 0.0
        self.adverse_wind_steps = 0
        self.best_horizontal_distance = float(
            np.linalg.norm((target[:2] - position[:2]).astype(np.float64))
        )
        self.consecutive_favorable_wind = 0
        self.consecutive_adverse_wind = 0
        self.consecutive_negative_horizontal_progress = 0
        self.idle_action_streak = 0
        self.z_window = deque(maxlen=max(1, z_window_maxlen))
        self._reset_z_window(float(position[2]))

    def _reset_z_window(self, z: float) -> None:
        self.z_window.clear()
        self.z_window_sum = z
        self.z_window_sumsq = z * z
        self.z_window.append(z)

    def append_z(self, z: float) -> None:
        if len(self.z_window) == self.z_window.maxlen:
            old = self.z_window[0]
            self.z_window_sum -= old
            self.z_window_sumsq -= old * old
        self.z_window.append(z)
        self.z_window_sum += z
        self.z_window_sumsq += z * z

    @staticmethod
    def window_std(sum_z: float, sumsq_z: float, count: int) -> float:
        if count <= 0:
            return 0.0
        mean = sum_z / count
        var = sumsq_z / count - mean * mean
        return float(np.sqrt(max(0.0, var)))
