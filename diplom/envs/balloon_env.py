from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from diplom.config import BalloonConfig, EnvironmentConfig, SimulationConfig
from diplom.sim.factory import create_simulation
from diplom.sim.simulation import SimResult, Simulation
from diplom.world import WorldBounds, resolve_balloon_config
from diplom.train.trajectory_steps_io import EnvStepsWriter, EpisodeFileRef
from diplom.wind.interp import WindInterpolator


class BalloonEnv(gym.Env):
    """Gymnasium-среда управления стратостатом.

    Принимает готовый WindInterpolator и считается его единственным владельцем:
    метод close() закрывает интерполятор. Не передавайте один интерполятор
    в несколько сред одновременно.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 20}

    def __init__(
        self,
        config: EnvironmentConfig,
        wind_interp: WindInterpolator,
        env_idx: int | None = None,
    ) -> None:
        self.dt = float(config.dt)
        self.initial_air_weight = config.initial_air_weight
        self.max_episode_steps = config.max_episode_steps
        self.randomize_start_state = config.randomize_start_state
        self.randomize_start_time = config.randomize_start_time
        self.train_start_time_delta = config.train_start_time_delta
        self.train_initial_position_delta = np.array(config.train_initial_position_delta, dtype=np.float32)
        self.train_target_position_delta = np.array(config.train_target_position_delta, dtype=np.float32)
        self.action_limit = np.float32(config.action_limit)
        self.target_reach_radius = np.float32(config.target_reach_radius)
        self.wind_interp = wind_interp
        self.world_bounds: WorldBounds = wind_interp.world_bounds
        self.env_idx = env_idx
        self.render_mode = "ansi"
        self._step_count = 0
        self._pending_step: dict[str, Any] | None = None
        self.base_balloon = resolve_balloon_config(config.balloon, self.world_bounds)

        self._trajectory_max_history = max(1, int(config.trajectory_max_history))
        self._steps_writer: EnvStepsWriter | None = None
        self._episode_count = 0
        self._episode_history: list[EpisodeFileRef] = []
        if config.trajectory_steps_dir is not None:
            writer_idx = env_idx if env_idx is not None else 0
            self._steps_writer = EnvStepsWriter(Path(config.trajectory_steps_dir), writer_idx)
            self._steps_writer.open_current()

        # Плоский вектор наблюдений (19 float32):
        #   position(3) + target_position(3) + delta_position(3) + wind(3)
        #   + energy(1) + air_weight(1) + vertical_speed(1)
        #   + vertical_acceleration(1) + air_density(1) + temperature(1) + pressure(1)
        self.action_space = spaces.Box(low=-self.action_limit, high=self.action_limit, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        # На каждый эпизод создаём новое состояние, чтобы не переносить скрытые эффекты между reset().
        self._step_count = 0
        self._pending_step = None
        episode_balloon = self._episode_balloon()
        self.sim = self._make_sim(episode_balloon)
        obs = self.to_obs(self.sim.snapshot())
        return obs, {}

    def step(self, action):
        clipped_action = self._clip_action(action)
        previous_position = np.array(self.sim.position, dtype=np.float32)

        self._step_count += 1
        result = self.sim.step(self.dt, clipped_action)

        previous_distance = float(np.linalg.norm(result.target_position - previous_position))
        current_distance = float(np.linalg.norm(result.target_position - result.position))
        progress_reward = previous_distance - current_distance

        reward = progress_reward
        terminated = bool(current_distance <= float(self.target_reach_radius))
        # Эпизод принудительно завершается по достижению лимита шагов.
        truncated = bool(self._step_count >= self.max_episode_steps)

        if terminated:
            reward += 100.0

        step_record = self._build_step_record(
            result=result,
            clipped_action=clipped_action,
            progress_reward=progress_reward,
            current_distance=current_distance,
            terminated=terminated,
            truncated=truncated,
        )
        self._pending_step = step_record
        if self._steps_writer is not None:
            self._steps_writer.append_step(step_record)
            if terminated or truncated:
                self._finalize_trajectory_episode(step_record)
        info = {
            "progress_reward": float(progress_reward),
            "distance_to_target": float(current_distance),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }

        return self.to_obs(result), float(reward), terminated, truncated, info

    def consume_step_record(self) -> dict[str, Any]:
        """Отдать запись шага (rollout/отладка) и очистить буфер.

        В ``info`` остаются только скаляры для TensorBoard; полные данные шага
        забираются здесь, чтобы ``DummyVecEnv`` не делал deepcopy numpy-массивов.
        При обучении JSONL пишется в subprocess через ``_steps_writer``.
        """
        if self._pending_step is None:
            return {}
        record = self._pending_step
        self._pending_step = None
        return record

    def get_trajectory_viz_state(self) -> dict[str, Any]:
        """Метаданные траекторий для снапшота рендера (один вызов за rollout)."""
        if self._steps_writer is None:
            return {}
        env_idx = self.env_idx if self.env_idx is not None else 0
        return {
            "env_idx": env_idx,
            "episode_count": self._episode_count,
            "history": list(self._episode_history),
            "current_steps_path": self._steps_writer.current_path,
            "current_step_count": self._steps_writer.step_count,
        }

    def _finalize_trajectory_episode(self, last_record: dict[str, Any]) -> None:
        if self._steps_writer is None or self._steps_writer.step_count == 0:
            return

        self._episode_count += 1
        ep_num = self._episode_count
        terminated = bool(last_record.get("terminated", False))
        outcome = "успех" if terminated else "truncated"
        step_count = self._steps_writer.step_count
        steps_path = self._steps_writer.finalize_episode(ep_num)
        target = tuple(float(v) for v in last_record.get("target_position", [0.0, 0.0, 0.0]))
        env_idx = self.env_idx if self.env_idx is not None else 0
        episode_ref = EpisodeFileRef(
            steps_path=steps_path,
            env_idx=env_idx,
            target_position=target,
            label=f"ep {ep_num} ({outcome}, {step_count} шагов)",
            step_count=step_count,
        )
        self._episode_history.append(episode_ref)
        if len(self._episode_history) > self._trajectory_max_history:
            old_ref = self._episode_history.pop(0)
            old_ref.steps_path.unlink(missing_ok=True)

    def _build_step_record(
        self,
        *,
        result: SimResult,
        clipped_action: float,
        progress_reward: float,
        current_distance: float,
        terminated: bool,
        truncated: bool,
    ) -> dict[str, Any]:
        return {
            "position": [float(v) for v in result.position],
            "wind": [float(v) for v in result.wind],
            "target_position": [float(v) for v in result.target_position],
            "action": float(clipped_action),
            "reward": float(progress_reward),
            "distance_to_target": float(current_distance),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "sim_time": str(self.sim.sim_time),
        }

    def close(self) -> None:
        if self._steps_writer is not None:
            self._steps_writer.close()
            self._steps_writer = None
        self.wind_interp.close()

    def render(self):
        """Текстовый снимок состояния для CLI/отладки."""
        snapshot = self.sim.snapshot()
        position = np.round(snapshot.position, 2)
        target_position = np.round(snapshot.target_position, 2)
        wind = np.round(snapshot.wind, 2)
        return f"pos={position.tolist()} target={target_position.tolist()} wind={wind.tolist()}"

    def _clip_action(self, action) -> float:
        value = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        limit = float(self.action_limit)
        if value < -limit:
            return -limit
        if value > limit:
            return limit
        return value

    def to_obs(self, sim_result: SimResult) -> np.ndarray:
        """Плоский вектор наблюдений shape=(19,) dtype=float32."""
        return np.concatenate(
            [
                sim_result.position,                                           # 3
                sim_result.target_position,                                    # 3
                sim_result.target_position - sim_result.position,             # 3
                sim_result.wind,                                               # 3
                [sim_result.energy_spent],                                     # 1
                [sim_result.air_weight],                                       # 1
                [sim_result.vertical_speed],                                   # 1
                [sim_result.vertical_acceleration],                            # 1
                [sim_result.air_density],                                      # 1
                [sim_result.temperature],                                      # 1
                [sim_result.pressure],                                         # 1
            ],
            dtype=np.float32,
        )

    def _make_sim(self, balloon: BalloonConfig) -> Simulation:
        return create_simulation(
            SimulationConfig(balloon=balloon, initial_air_weight=self.initial_air_weight),
            self.wind_interp,
            env_idx=self.env_idx,
        )

    def _episode_balloon(self) -> BalloonConfig:
        initial_position = self.base_balloon.initial_position
        target_position = self.base_balloon.target_position
        sim_time = self.base_balloon.sim_time

        if self.randomize_start_state:
            # Для train-режима рандомизируем старт вокруг центра,
            # а цель выбираем по всему миру, гарантируя минимальное расстояние.
            initial_position = self._sample_position(
                self.base_balloon.initial_position,
                self.train_initial_position_delta,
            )
            target_position = self._sample_target_position(initial_position)

        if self.randomize_start_time:
            # Время эпизода выбираем вокруг середины диапазона датасета и ограничиваем его границами.
            sim_time = self._sample_time(
                self.wind_interp.time_min,
                self.wind_interp.time_max,
                self.train_start_time_delta,
            )

        return BalloonConfig(
            initial_position=initial_position,
            target_position=target_position,
            sim_time=sim_time,
        )

    def _sample_target_position(self, initial_position: np.ndarray) -> np.ndarray:
        """Сэмплирует целевую позицию с учётом train_target_position_delta и
        гарантирует минимальное расстояние до стартовой точки."""
        min_distance = 3000.0

        while True:
            candidate = self._sample_position(self.base_balloon.target_position, self.train_target_position_delta)
            if float(np.linalg.norm(candidate - initial_position)) >= min_distance:
                return candidate


    def _sample_position(self, center: np.ndarray, delta: np.ndarray) -> np.ndarray:
        low = np.array(
            [self.world_bounds.x_min, self.world_bounds.y_min, self.world_bounds.z_min],
            dtype=np.float32,
        )
        high = np.array(
            [self.world_bounds.x_max, self.world_bounds.y_max, self.world_bounds.z_max],
            dtype=np.float32,
        )

        sample_low = np.maximum(center - delta, low)
        sample_high = np.minimum(center + delta, high)
        return self.np_random.uniform(low=sample_low, high=sample_high).astype(np.float32)

    def _sample_time(
        self,
        time_min: np.datetime64,
        time_max: np.datetime64,
        delta: np.timedelta64,
    ) -> np.datetime64:
        min_ns = np.datetime64(time_min, "ns").astype(np.int64)
        max_ns = np.datetime64(time_max, "ns").astype(np.int64)
        if max_ns <= min_ns:
            return np.datetime64(min_ns, "ns")

        mid_ns = min_ns + (max_ns - min_ns) // 2
        delta_ns = np.asarray(delta, dtype="timedelta64[ns]").astype(np.int64)
        low_ns = max(min_ns, mid_ns - delta_ns)
        high_ns = min(max_ns, mid_ns + delta_ns)
        if high_ns <= low_ns:
            return np.datetime64(mid_ns, "ns")

        sampled_ns = int(self.np_random.integers(low_ns, high_ns + 1))
        return np.datetime64(sampled_ns, "ns")
