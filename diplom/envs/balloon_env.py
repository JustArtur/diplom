from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from diplom.config import BalloonConfig, EnvironmentConfig, SimulationConfig
from diplom.envs.constants import TARGET_VERTICAL_REACH_RADIUS
from diplom.envs.observations import ObsStepContext, get_obs_spec
from diplom.envs.rewards import RewardState, RewardStepContext
from diplom.envs.wind_probes import compute_probe_winds, should_compute_probe_winds
from diplom.sim.factory import create_simulation
from diplom.sim.simulation import SimResult, Simulation
from diplom.world import WorldBounds, resolve_balloon_config
from diplom.trajectory.steps_io import (
    EnvStepsWriter,
    EpisodeFileRef,
    archive_success_episode,
)
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
        self._prev_z = 0.0
        self.reward_name = config.reward_name
        self.obs_name = config.obs_name
        self._reward_module = import_module(f"diplom.envs.rewards.{config.reward_name}")
        self._build_obs, self._obs_dim = get_obs_spec(config.obs_name)
        self._reward_state = RewardState()
        self._compute_reward = self._reward_module.compute_reward
        self._wind_align_scale = float(getattr(self._reward_module, "WIND_ALIGN_SCALE", 20.0))
        self._z_stick_window_steps = int(getattr(self._reward_module, "Z_STICK_WINDOW_STEPS", 50_000))
        self._obs_needs_probe_winds = config.obs_name == "default"
        self._reward_needs_probe_winds = bool(getattr(self._reward_module, "NEEDS_PROBE_WINDS", False))
        self.normalize_observations = bool(config.normalize_observations)
        self.randomize_start_state = config.randomize_start_state
        self.train_initial_position_delta = np.array(config.train_initial_position_delta, dtype=np.float32)
        self.train_target_position_delta = np.array(config.train_target_position_delta, dtype=np.float32)
        self.action_limit = np.float32(config.action_limit)
        self.target_reach_radius = np.float32(config.target_reach_radius)
        self.target_vertical_reach_radius = np.float32(TARGET_VERTICAL_REACH_RADIUS)
        self.wind_interp = wind_interp
        self.world_bounds: WorldBounds = wind_interp.world_bounds
        self.env_idx = env_idx
        self.render_mode = "ansi"
        self._step_count = 0
        self._pending_step: dict[str, Any] | None = None
        self._pending_step_build: dict[str, Any] | None = None
        self.base_balloon = resolve_balloon_config(config.balloon, self.world_bounds)

        self._trajectory_max_history = max(1, int(config.trajectory_max_history))
        self._steps_writer: EnvStepsWriter | None = None
        self._episode_count = 0
        self._episode_history: list[EpisodeFileRef] = []
        if config.trajectory_steps_dir is not None:
            writer_idx = env_idx if env_idx is not None else 0
            self._steps_writer = EnvStepsWriter(Path(config.trajectory_steps_dir), writer_idx)
            self._steps_writer.open_current()

        self.action_space = spaces.Box(low=-self.action_limit, high=self.action_limit, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        # На каждый эпизод создаём новое состояние, чтобы не переносить скрытые эффекты между reset().
        self._step_count = 0
        self._pending_step = None
        self._pending_step_build = None
        episode_balloon = self._episode_balloon()
        self.sim = self._make_sim(episode_balloon)
        snapshot = self.sim.snapshot()
        self._init_episode_reward_state(snapshot)
        obs = self.to_obs(snapshot, previous_position=None)
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
        self._reward_state.reset(
            target=target,
            position=position,
            wind=wind,
            wind_toward_fn=self._wind_toward_target,
            z_window_maxlen=self._z_stick_window_steps,
        )

    def step(self, action):
        clipped_action = self._clip_action(action)
        previous_position = np.array(self.sim.position, dtype=np.float32)
        previous_energy = float(self.sim.energy_spent)

        self._step_count += 1
        result = self.sim.step(self.dt, clipped_action)
        target = np.array(result.target_position, dtype=np.float32)
        current_position = np.array(result.position, dtype=np.float32)

        curr_horizontal = self._horizontal_distance(target, current_position)
        current_distance = float(np.linalg.norm(target - current_position))
        energy_delta = max(0.0, float(result.energy_spent) - previous_energy)

        probe_winds, max_probe_wind_toward = self._maybe_compute_probe_winds(
            result=result,
            previous_position=previous_position,
        )

        reward_result = self._compute_reward(
            self.wind_interp,
            result,
            RewardStepContext(
                result=result,
                previous_position=previous_position,
                clipped_action=clipped_action,
                energy_delta=energy_delta,
                sim_time=self.sim.sim_time,
                z_min=float(self.world_bounds.z_min),
                z_max=float(self.world_bounds.z_max),
                max_probe_wind_toward=max_probe_wind_toward,
                boundary_contact=bool(self.sim.last_step_boundary_contact),
                step_count=self._step_count,
                max_episode_steps=self.max_episode_steps,
                target_reach_radius=float(self.target_reach_radius),
                target_vertical_reach_radius=float(self.target_vertical_reach_radius),
            ),
            self._reward_state,
        )

        reward = reward_result.reward
        terminated = reward_result.terminated
        truncated = reward_result.truncated
        horizontal_progress = reward_result.horizontal_progress
        vertical_progress = reward_result.vertical_progress
        wind_toward = reward_result.wind_toward
        wind_align_delta = reward_result.wind_align_delta
        terms = reward_result.terms
        progress_reward = horizontal_progress + vertical_progress

        step_record_kwargs = {
            "result": result,
            "clipped_action": clipped_action,
            "progress_reward": progress_reward,
            "horizontal_progress": horizontal_progress,
            "vertical_progress": vertical_progress,
            "wind_toward": wind_toward,
            "wind_align_delta": wind_align_delta,
            "current_distance": current_distance,
            "horizontal_distance": curr_horizontal,
            "reward": float(reward),
            "wind_align_term": terms["reward_wind_align_term"],
            "wind_align_delta_term": terms["reward_wind_align_delta_term"],
            "progress_term": terms["reward_progress_term"],
            "goal_term": terms.get("reward_goal_term", 0.0),
            "distance_term": terms["reward_distance_term"],
            "energy_term": terms["reward_energy_term"],
            "boundary_term": terms["reward_boundary_term"],
            "best_distance_term": terms["reward_best_distance_term"],
            "regression_term": terms["reward_distance_regression_term"],
            "hold_close_term": terms["reward_hold_close_term"],
            "wind_streak_term": terms["reward_wind_streak_term"],
            "wind_adverse_streak_term": terms["reward_wind_adverse_streak_term"],
            "wind_scan_term": terms["reward_wind_scan_term"],
            "adverse_wind_close_term": terms["reward_adverse_wind_close_term"],
            "high_altitude_term": terms["reward_high_altitude_term"],
            "idle_action_term": terms["reward_idle_action_term"],
            "z_stick_term": terms["reward_z_stick_term"],
            "consecutive_favorable_wind": reward_result.consecutive_favorable_wind,
            "consecutive_adverse_wind": reward_result.consecutive_adverse_wind,
            "terminated": terminated,
            "truncated": truncated,
        }
        if self._steps_writer is not None:
            step_record = self._build_step_record(**step_record_kwargs)
            self._pending_step = step_record
            self._pending_step_build = None
            self._steps_writer.append_step(step_record)
            if terminated or truncated:
                self._finalize_trajectory_episode(step_record)
        else:
            self._pending_step = None
            self._pending_step_build = step_record_kwargs
        info = {
            "progress_reward": float(progress_reward),
            "horizontal_progress": float(horizontal_progress),
            "distance_to_target": float(current_distance),
            "horizontal_distance": float(curr_horizontal),
            "wind_toward": float(wind_toward),
            "wind_align_delta": float(wind_align_delta),
            **{key: float(value) for key, value in terms.items()},
            "consecutive_favorable_wind": float(reward_result.consecutive_favorable_wind),
            "consecutive_adverse_wind": float(reward_result.consecutive_adverse_wind),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }

        return self.to_obs(result, previous_position=previous_position, probe_winds=probe_winds), float(reward), terminated, truncated, info

    def _maybe_compute_probe_winds(
        self,
        *,
        result: SimResult,
        previous_position: np.ndarray | None,
    ) -> tuple[np.ndarray | None, float | None]:
        position = np.asarray(result.position, dtype=np.float32)
        if not should_compute_probe_winds(
            obs_needs_probes=self._obs_needs_probe_winds,
            reward_needs_probes=self._reward_needs_probe_winds,
            previous_position=previous_position,
            current_position=position,
        ):
            return None, None
        probe_winds, max_probe = compute_probe_winds(
            self.wind_interp,
            position=position,
            target=np.asarray(result.target_position, dtype=np.float32),
            sim_time=self.sim.sim_time,
            z_min=float(self.world_bounds.z_min),
            z_max=float(self.world_bounds.z_max),
        )
        return probe_winds, max_probe

    def to_obs(
        self,
        sim_result: SimResult,
        *,
        previous_position: np.ndarray | None = None,
        probe_winds: np.ndarray | None = None,
    ) -> np.ndarray:
        if probe_winds is None and self._obs_needs_probe_winds:
            probe_winds, _ = self._maybe_compute_probe_winds(
                result=sim_result,
                previous_position=previous_position,
            )
        return self._build_obs(
            self.wind_interp,
            sim_result,
            ObsStepContext(
                sim_time=self.sim.sim_time,
                z_min=float(self.world_bounds.z_min),
                z_max=float(self.world_bounds.z_max),
                normalize=self.normalize_observations,
                reward_state=self._reward_state,
                wind_align_scale=self._wind_align_scale,
                probe_winds=probe_winds,
            ),
        )

    def consume_step_record(self) -> dict[str, Any]:
        """Отдать запись шага (rollout/отладка) и очистить буфер.

        В ``info`` остаются только скаляры для TensorBoard; полные данные шага
        забираются здесь, чтобы ``DummyVecEnv`` не делал deepcopy numpy-массивов.
        При обучении JSONL пишется в subprocess через ``_steps_writer``.
        """
        if self._pending_step is None and self._pending_step_build is not None:
            self._pending_step = self._build_step_record(**self._pending_step_build)
            self._pending_step_build = None
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
            "current_steps_path": self._steps_writer.current_path.resolve(),
            "current_step_count": self._steps_writer.step_count,
        }

    def _trim_episode_history(self) -> None:
        # Файлы не удаляем здесь: рендер читает их асинхронно из снапшота,
        # и удаление до завершения рендера оставляет в HTML только текущий эпизод.
        # Временные JSONL убираются в cleanup_steps_dir() после остановки обучения.
        while len(self._episode_history) > self._trajectory_max_history:
            self._episode_history.pop(0)

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
        success = terminated
        if success:
            archived_path = archive_success_episode(
                steps_path,
                self._steps_writer.output_dir,
                env_idx=env_idx,
                episode_num=ep_num,
                step_count=step_count,
            )
            steps_path.unlink(missing_ok=True)
            steps_path = archived_path

        episode_ref = EpisodeFileRef(
            steps_path=Path(steps_path).resolve(),
            env_idx=env_idx,
            target_position=target,
            label=f"ep {ep_num} ({outcome}, {step_count} шагов)",
            step_count=step_count,
            success=success,
        )
        self._episode_history.append(episode_ref)
        self._trim_episode_history()

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
        goal_term: float,
        distance_term: float,
        energy_term: float,
        boundary_term: float,
        best_distance_term: float,
        regression_term: float,
        hold_close_term: float,
        wind_streak_term: float,
        wind_adverse_streak_term: float,
        wind_scan_term: float,
        adverse_wind_close_term: float,
        high_altitude_term: float,
        idle_action_term: float,
        z_stick_term: float,
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
            "reward_goal_term": goal_term,
            "reward_distance_term": distance_term,
            "reward_energy_term": energy_term,
            "reward_boundary_term": boundary_term,
            "reward_best_distance_term": best_distance_term,
            "reward_distance_regression_term": regression_term,
            "reward_hold_close_term": hold_close_term,
            "reward_wind_streak_term": wind_streak_term,
            "reward_wind_adverse_streak_term": wind_adverse_streak_term,
            "reward_wind_scan_term": wind_scan_term,
            "reward_adverse_wind_close_term": adverse_wind_close_term,
            "reward_high_altitude_term": high_altitude_term,
            "reward_idle_action_term": idle_action_term,
            "reward_z_stick_term": z_stick_term,
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

    def _make_sim(self, balloon: BalloonConfig) -> Simulation:
        return create_simulation(
            SimulationConfig(balloon=balloon, initial_air_weight=self.initial_air_weight),
            self.wind_interp,
            env_idx=self.env_idx,
        )

    def _episode_balloon(self) -> BalloonConfig:
        initial_position = self.base_balloon.initial_position
        target_position = self.base_balloon.target_position

        if self.randomize_start_state:
            # Для train-режима рандомизируем старт по X/Y; высота — всегда base_balloon.initial_position[2].
            initial_position = self._sample_initial_position(
                self.base_balloon.initial_position,
                self.train_initial_position_delta,
            )
            target_position = self._sample_target_position(initial_position)

        return BalloonConfig(
            initial_position=initial_position,
            target_position=target_position,
            sim_time=self.base_balloon.sim_time,
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
