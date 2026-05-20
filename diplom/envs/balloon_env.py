from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from diplom.config import BalloonConfig, EnvironmentConfig, SimulationConfig
from diplom.envs.constants import (
    OBS_AIR_DENSITY_SCALE,
    OBS_AIR_WEIGHT_SCALE,
    OBS_ALTITUDE_SCALE,
    OBS_ENERGY_SCALE,
    OBS_PRESSURE_SCALE,
    OBS_TEMPERATURE_SCALE,
    OBS_VERTICAL_ACCELERATION_SCALE,
    OBS_VERTICAL_SPEED_SCALE,
    OBS_WIND_SCALE,
    OBS_XY_SCALE,
    REWARD_WIND_ALIGN_SCALE,
)
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
        self.reward_wind_align_scale = float(config.reward_wind_align_scale)
        self.reward_wind_align_coef = float(config.reward_wind_align_coef)
        self.reward_wind_align_delta_coef = float(config.reward_wind_align_delta_coef)
        self.reward_horizontal_progress_scale = float(config.reward_horizontal_progress_scale)
        self.reward_horizontal_progress_pos_coef = float(config.reward_horizontal_progress_pos_coef)
        self.reward_horizontal_progress_neg_coef = float(config.reward_horizontal_progress_neg_coef)
        self.reward_vertical_progress_scale = float(config.reward_vertical_progress_scale)
        self.reward_vertical_progress_pos_coef = float(config.reward_vertical_progress_pos_coef)
        self.reward_vertical_progress_neg_coef = float(config.reward_vertical_progress_neg_coef)
        self.reward_horizontal_distance_scale = float(config.reward_horizontal_distance_scale)
        self.reward_horizontal_distance_coef = float(config.reward_horizontal_distance_coef)
        self.reward_best_distance_bonus = float(config.reward_best_distance_bonus)
        self.reward_energy_coef = float(config.reward_energy_coef)
        self.reward_energy_scale = float(config.reward_energy_scale)
        self.reward_boundary_penalty = float(config.reward_boundary_penalty)
        self.reward_wind_favorable_threshold = float(config.reward_wind_favorable_threshold)
        self.reward_wind_adverse_threshold = float(config.reward_wind_adverse_threshold)
        self.reward_wind_favorable_streak_steps = int(config.reward_wind_favorable_streak_steps)
        self.reward_wind_adverse_streak_steps = int(config.reward_wind_adverse_streak_steps)
        self.reward_wind_favorable_streak_bonus = float(config.reward_wind_favorable_streak_bonus)
        self.reward_wind_adverse_streak_penalty = float(config.reward_wind_adverse_streak_penalty)
        self.success_reward = float(config.success_reward)
        self._prev_wind_toward = 0.0
        self._best_horizontal_distance = float("inf")
        self._consecutive_favorable_wind = 0
        self._consecutive_adverse_wind = 0
        self.normalize_observations = bool(config.normalize_observations)
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

        # Плоский вектор наблюдений (20 float32):
        #   position(3) + target_position(3) + delta_position(3) + wind(3)
        #   + energy(1) + air_weight(1) + vertical_speed(1)
        #   + vertical_acceleration(1) + air_density(1) + temperature(1) + pressure(1)
        #   + wind_toward(1)
        self.action_space = spaces.Box(low=-self.action_limit, high=self.action_limit, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)

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
        snapshot = self.sim.snapshot()
        self._init_episode_reward_state(snapshot)
        obs = self.to_obs(snapshot)
        return obs, {}

    @staticmethod
    def _horizontal_distance(target: np.ndarray, position: np.ndarray) -> float:
        delta = target[:2] - position[:2]
        return float(np.linalg.norm(delta))

    @staticmethod
    def _vertical_distance(target: np.ndarray, position: np.ndarray) -> float:
        return float(abs(target[2] - position[2]))

    @staticmethod
    def _wind_toward_target(target: np.ndarray, position: np.ndarray, wind: np.ndarray) -> float:
        delta_xy = target[:2] - position[:2]
        norm = float(np.linalg.norm(delta_xy))
        if norm < 1e-6:
            return 0.0
        unit = delta_xy / norm
        return float(wind[0] * unit[0] + wind[1] * unit[1])

    def _init_episode_reward_state(self, snapshot: SimResult) -> None:
        target = np.asarray(snapshot.target_position, dtype=np.float32)
        position = np.asarray(snapshot.position, dtype=np.float32)
        wind = np.asarray(snapshot.wind, dtype=np.float32)
        self._prev_wind_toward = self._wind_toward_target(target, position, wind)
        self._best_horizontal_distance = self._horizontal_distance(target, position)
        self._consecutive_favorable_wind = 0
        self._consecutive_adverse_wind = 0

    def _wind_streak_terms(self, wind_toward: float) -> tuple[float, float]:
        """Бонус за удержание попутного ветра; штраф за долгое «застревание» во встречном слое."""
        if wind_toward > self.reward_wind_favorable_threshold:
            self._consecutive_favorable_wind += 1
            self._consecutive_adverse_wind = 0
        elif wind_toward < self.reward_wind_adverse_threshold:
            self._consecutive_adverse_wind += 1
            self._consecutive_favorable_wind = 0
        else:
            self._consecutive_favorable_wind = 0
            self._consecutive_adverse_wind = 0

        streak_term = 0.0
        if self._consecutive_favorable_wind >= self.reward_wind_favorable_streak_steps:
            streak_term = self.reward_wind_favorable_streak_bonus

        adverse_streak_term = 0.0
        if self._consecutive_adverse_wind >= self.reward_wind_adverse_streak_steps:
            adverse_streak_term = -self.reward_wind_adverse_streak_penalty

        return streak_term, adverse_streak_term

    @staticmethod
    def _asymmetric_progress_term(
        progress: float,
        scale: float,
        pos_coef: float,
        neg_coef: float,
    ) -> float:
        if progress >= 0.0:
            return pos_coef * progress / scale
        return neg_coef * progress / scale

    def step(self, action):
        clipped_action = self._clip_action(action)
        previous_position = np.array(self.sim.position, dtype=np.float32)
        previous_energy = float(self.sim.energy_spent)

        self._step_count += 1
        result = self.sim.step(self.dt, clipped_action)
        current_position = np.array(result.position, dtype=np.float32)
        target = np.array(result.target_position, dtype=np.float32)
        wind = np.array(result.wind, dtype=np.float32)

        prev_horizontal = self._horizontal_distance(target, previous_position)
        curr_horizontal = self._horizontal_distance(target, current_position)
        horizontal_progress = prev_horizontal - curr_horizontal

        prev_vertical = self._vertical_distance(target, previous_position)
        curr_vertical = self._vertical_distance(target, current_position)
        vertical_progress = prev_vertical - curr_vertical

        current_distance = float(np.linalg.norm(target - current_position))
        energy_delta = max(0.0, float(result.energy_spent) - previous_energy)

        wind_toward = self._wind_toward_target(target, current_position, wind)
        wind_align_delta = wind_toward - self._prev_wind_toward

        wind_align_term = self.reward_wind_align_coef * wind_toward / self.reward_wind_align_scale
        wind_align_delta_term = (
            self.reward_wind_align_delta_coef * wind_align_delta / self.reward_wind_align_scale
        )
        progress_term = self._asymmetric_progress_term(
            horizontal_progress,
            self.reward_horizontal_progress_scale,
            self.reward_horizontal_progress_pos_coef,
            self.reward_horizontal_progress_neg_coef,
        ) + self._asymmetric_progress_term(
            vertical_progress,
            self.reward_vertical_progress_scale,
            self.reward_vertical_progress_pos_coef,
            self.reward_vertical_progress_neg_coef,
        )
        distance_term = (
            -self.reward_horizontal_distance_coef * curr_horizontal / self.reward_horizontal_distance_scale
        )
        energy_term = -self.reward_energy_coef * energy_delta / self.reward_energy_scale
        boundary_term = (
            -self.reward_boundary_penalty if self.sim.last_step_boundary_contact else 0.0
        )
        best_distance_term = 0.0
        if curr_horizontal < self._best_horizontal_distance:
            best_distance_term = self.reward_best_distance_bonus
            self._best_horizontal_distance = curr_horizontal

        wind_streak_term, wind_adverse_streak_term = self._wind_streak_terms(wind_toward)

        reward = (
            wind_align_term
            + wind_align_delta_term
            + progress_term
            + distance_term
            + energy_term
            + boundary_term
            + best_distance_term
            + wind_streak_term
            + wind_adverse_streak_term
        )

        terminated = bool(current_distance <= float(self.target_reach_radius))
        truncated = bool(self._step_count >= self.max_episode_steps)

        if terminated:
            reward += self.success_reward

        self._prev_wind_toward = wind_toward

        progress_reward = horizontal_progress + vertical_progress

        step_record = self._build_step_record(
            result=result,
            clipped_action=clipped_action,
            progress_reward=progress_reward,
            horizontal_progress=horizontal_progress,
            vertical_progress=vertical_progress,
            wind_toward=wind_toward,
            wind_align_delta=wind_align_delta,
            current_distance=current_distance,
            horizontal_distance=curr_horizontal,
            reward=float(reward),
            wind_align_term=wind_align_term,
            wind_align_delta_term=wind_align_delta_term,
            progress_term=progress_term,
            distance_term=distance_term,
            energy_term=energy_term,
            boundary_term=boundary_term,
            best_distance_term=best_distance_term,
            wind_streak_term=wind_streak_term,
            wind_adverse_streak_term=wind_adverse_streak_term,
            consecutive_favorable_wind=self._consecutive_favorable_wind,
            consecutive_adverse_wind=self._consecutive_adverse_wind,
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
            "horizontal_progress": float(horizontal_progress),
            "distance_to_target": float(current_distance),
            "horizontal_distance": float(curr_horizontal),
            "wind_toward": float(wind_toward),
            "wind_align_delta": float(wind_align_delta),
            "reward_wind_align_term": float(wind_align_term),
            "reward_wind_align_delta_term": float(wind_align_delta_term),
            "reward_progress_term": float(progress_term),
            "reward_distance_term": float(distance_term),
            "reward_energy_term": float(energy_term),
            "reward_boundary_term": float(boundary_term),
            "reward_best_distance_term": float(best_distance_term),
            "reward_wind_streak_term": float(wind_streak_term),
            "reward_wind_adverse_streak_term": float(wind_adverse_streak_term),
            "consecutive_favorable_wind": float(self._consecutive_favorable_wind),
            "consecutive_adverse_wind": float(self._consecutive_adverse_wind),
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
        horizontal_progress: float,
        vertical_progress: float,
        wind_toward: float,
        wind_align_delta: float,
        current_distance: float,
        horizontal_distance: float,
        reward: float,
        wind_align_term: float,
        wind_align_delta_term: float,
        progress_term: float,
        distance_term: float,
        energy_term: float,
        boundary_term: float,
        best_distance_term: float,
        wind_streak_term: float,
        wind_adverse_streak_term: float,
        consecutive_favorable_wind: int,
        consecutive_adverse_wind: int,
        terminated: bool,
        truncated: bool,
    ) -> dict[str, Any]:
        return {
            "position": [float(v) for v in result.position],
            "wind": [float(v) for v in result.wind],
            "target_position": [float(v) for v in result.target_position],
            "action": float(clipped_action),
            "reward": reward,
            "progress_reward": float(progress_reward),
            "horizontal_progress": float(horizontal_progress),
            "vertical_progress": float(vertical_progress),
            "wind_toward": float(wind_toward),
            "wind_align_delta": float(wind_align_delta),
            "reward_wind_align_term": wind_align_term,
            "reward_wind_align_delta_term": wind_align_delta_term,
            "reward_progress_term": progress_term,
            "reward_distance_term": distance_term,
            "reward_energy_term": energy_term,
            "reward_boundary_term": boundary_term,
            "reward_best_distance_term": best_distance_term,
            "reward_wind_streak_term": wind_streak_term,
            "reward_wind_adverse_streak_term": wind_adverse_streak_term,
            "consecutive_favorable_wind": int(consecutive_favorable_wind),
            "consecutive_adverse_wind": int(consecutive_adverse_wind),
            "distance_to_target": float(current_distance),
            "horizontal_distance": float(horizontal_distance),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "sim_time": str(self.sim.sim_time),
            "vertical_speed": float(result.vertical_speed),
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
        """Плоский вектор наблюдений shape=(20,) dtype=float32."""
        position = np.asarray(sim_result.position, dtype=np.float32)
        target = np.asarray(sim_result.target_position, dtype=np.float32)
        delta = target - position
        wind = np.asarray(sim_result.wind, dtype=np.float32)
        wind_toward = self._wind_toward_target(target, position, wind)
        if not self.normalize_observations:
            return np.concatenate(
                [
                    position,
                    target,
                    delta,
                    wind,
                    [sim_result.energy_spent],
                    [sim_result.air_weight],
                    [sim_result.vertical_speed],
                    [sim_result.vertical_acceleration],
                    [sim_result.air_density],
                    [sim_result.temperature],
                    [sim_result.pressure],
                    [wind_toward],
                ],
                dtype=np.float32,
            )

        def _scale_xyz(vec: np.ndarray, xy_scale: float, z_scale: float) -> np.ndarray:
            out = vec.astype(np.float32, copy=True)
            out[0] /= xy_scale
            out[1] /= xy_scale
            out[2] /= z_scale
            return out

        return np.concatenate(
            [
                _scale_xyz(position, OBS_XY_SCALE, OBS_ALTITUDE_SCALE),
                _scale_xyz(target, OBS_XY_SCALE, OBS_ALTITUDE_SCALE),
                _scale_xyz(delta, OBS_XY_SCALE, OBS_ALTITUDE_SCALE),
                wind / OBS_WIND_SCALE,
                [sim_result.energy_spent / OBS_ENERGY_SCALE],
                [sim_result.air_weight / OBS_AIR_WEIGHT_SCALE],
                [sim_result.vertical_speed / OBS_VERTICAL_SPEED_SCALE],
                [sim_result.vertical_acceleration / OBS_VERTICAL_ACCELERATION_SCALE],
                [sim_result.air_density / OBS_AIR_DENSITY_SCALE],
                [sim_result.temperature / OBS_TEMPERATURE_SCALE],
                [sim_result.pressure / OBS_PRESSURE_SCALE],
                [wind_toward / REWARD_WIND_ALIGN_SCALE],
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
            # Для train-режима рандомизируем старт по X/Y; высота — всегда base_balloon.initial_position[2].
            initial_position = self._sample_initial_position(
                self.base_balloon.initial_position,
                self.train_initial_position_delta,
            )
            target_position = self._sample_target_position(initial_position)

        if self.randomize_start_time:
            sim_time = self._sample_time_from_dataset_start(
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


    def _sample_initial_position(self, center: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """Сэмплирует стартовую позицию: X/Y в окне delta, Z фиксирована."""
        sample = self._sample_position(center, delta)
        sample[2] = np.float32(center[2])
        return sample

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

    def _sample_time_from_dataset_start(
        self,
        time_min: np.datetime64,
        time_max: np.datetime64,
        delta: np.timedelta64,
    ) -> np.datetime64:
        min_ns = np.datetime64(time_min, "ns").astype(np.int64)
        max_ns = np.datetime64(time_max, "ns").astype(np.int64)
        if max_ns <= min_ns:
            return np.datetime64(min_ns, "ns")

        delta_ns = np.asarray(delta, dtype="timedelta64[ns]").astype(np.int64)
        high_ns = min(max_ns, min_ns + delta_ns)
        if high_ns <= min_ns:
            return np.datetime64(min_ns, "ns")

        sampled_ns = int(self.np_random.integers(min_ns, high_ns + 1))
        return np.datetime64(sampled_ns, "ns")
