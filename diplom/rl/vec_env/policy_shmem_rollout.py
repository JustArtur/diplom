"""Гибрид п.3+п.4: policy+env в воркере, rollout-буфер через shared memory (без pickle чанка)."""

from __future__ import annotations

import multiprocessing as mp
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from stable_baselines3.common.vec_env.subproc_vec_env import _stack_obs

from diplom.rl.logging.env_info_log_keys import ENV_INFO_LOG_KEYS, INFO_KEY_INDEX, N_ENV_INFO_KEYS
from diplom.rl.vec_env.rollout_worker import build_worker_policy, prepare_env_action

_INFO_NAN = np.float32(np.nan)
_INFO_WRITE_ITEMS: tuple[tuple[int, str], ...] = tuple(
    (INFO_KEY_INDEX[key], key) for key in ENV_INFO_LOG_KEYS
)


def _create_shared_array(shape: tuple[int, ...], dtype: np.dtype) -> tuple[shared_memory.SharedMemory, np.ndarray]:
    dtype = np.dtype(dtype)
    size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    shm = shared_memory.SharedMemory(create=True, size=size)
    array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return shm, array


def _attach_shared_array(name: str, shape: tuple[int, ...], dtype: np.dtype) -> tuple[shared_memory.SharedMemory, np.ndarray]:
    shm = shared_memory.SharedMemory(name=name)
    array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return shm, array


@dataclass(frozen=True, slots=True)
class _RolloutShmemNames:
    observations: str
    actions: str
    rewards: str
    episode_starts: str
    dones: str
    values: str
    log_probs: str
    final_observations: str
    time_limit_truncated: str
    terminal_observations: str
    info_scalars: str
    episode_return: str
    episode_length: str
    episode_success: str


def _write_info_scalars(
    *,
    step_idx: int,
    env_idx: int,
    info: dict[str, Any],
    info_scalars: np.ndarray,
    episode_return: np.ndarray,
    episode_length: np.ndarray,
    episode_success: np.ndarray,
    time_limit_truncated: np.ndarray,
    terminal_observations: np.ndarray,
    done: bool,
    terminal_obs: Any,
) -> None:
    row = info_scalars[env_idx, step_idx]
    for idx, key in _INFO_WRITE_ITEMS:
        value = info.get(key)
        if value is not None:
            row[idx] = np.float32(value)
    episode = info.get("episode")
    if episode is not None:
        episode_return[env_idx, step_idx] = np.float32(episode["r"])
        episode_length[env_idx, step_idx] = np.float32(episode["l"])
        episode_success[env_idx, step_idx] = np.float32(1.0 if info.get("terminated") else 0.0)
    truncated = bool(info.get("TimeLimit.truncated", False))
    time_limit_truncated[env_idx, step_idx] = truncated
    if done and truncated:
        terminal_observations[env_idx, step_idx] = np.asarray(terminal_obs, dtype=np.float32)


def _collect_rollout_into_shmem(
    *,
    env: gym.Env,
    policy: Any,
    current_obs: np.ndarray,
    env_idx: int,
    n_steps: int,
    initial_episode_start: bool,
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    episode_starts: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    log_probs: np.ndarray,
    final_observations: np.ndarray,
    time_limit_truncated: np.ndarray,
    terminal_observations: np.ndarray,
    info_scalars: np.ndarray,
    episode_return: np.ndarray,
    episode_length: np.ndarray,
    episode_success: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    episode_start = initial_episode_start
    final_reset_info: dict[str, Any] = {}

    info_scalars[env_idx, :n_steps, :] = _INFO_NAN
    episode_return[env_idx, :n_steps] = _INFO_NAN
    episode_length[env_idx, :n_steps] = _INFO_NAN
    episode_success[env_idx, :n_steps] = _INFO_NAN

    with th.no_grad():
        for step_idx in range(n_steps):
            observations[env_idx, step_idx] = current_obs
            episode_starts[env_idx, step_idx] = episode_start

            obs_tensor = obs_as_tensor(np.expand_dims(current_obs, axis=0), device="cpu")
            action_tensor, value_tensor, log_prob_tensor = policy(obs_tensor)
            raw_action = action_tensor.cpu().numpy()
            if raw_action.ndim > len(act_shape):
                raw_action = raw_action[0]
            actions[env_idx, step_idx] = np.asarray(raw_action, dtype=np.float32).reshape(act_shape)
            values[env_idx, step_idx] = np.float32(value_tensor.cpu().numpy().reshape(-1)[0])
            log_probs[env_idx, step_idx] = np.float32(log_prob_tensor.cpu().numpy().reshape(-1)[0])

            env_action = prepare_env_action(policy, env.action_space, actions[env_idx, step_idx])
            new_obs, reward, terminated, truncated, info = env.step(env_action)
            done = bool(terminated or truncated)
            dones[env_idx, step_idx] = done
            info["TimeLimit.truncated"] = bool(truncated and not terminated)
            terminal_obs = new_obs
            if done:
                info["terminal_observation"] = new_obs
                new_obs, final_reset_info = env.reset()

            rewards[env_idx, step_idx] = np.float32(reward)
            _write_info_scalars(
                step_idx=step_idx,
                env_idx=env_idx,
                info=info,
                info_scalars=info_scalars,
                episode_return=episode_return,
                episode_length=episode_length,
                episode_success=episode_success,
                time_limit_truncated=time_limit_truncated,
                terminal_observations=terminal_observations,
                done=done,
                terminal_obs=terminal_obs,
            )
            current_obs = np.asarray(new_obs, dtype=np.float32)
            episode_start = done

    final_observations[env_idx] = current_obs
    return current_obs, final_reset_info


def _policy_shmem_rollout_worker(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
    env_idx: int,
    rollout_steps: int,
    shmem_names: _RolloutShmemNames,
    n_envs: int,
    obs_shape: tuple[int, ...],
    act_shape: tuple[int, ...],
) -> None:
    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()
    obs_steps_shape = (n_envs, rollout_steps, *obs_shape)
    act_steps_shape = (n_envs, rollout_steps, *act_shape)
    term_obs_shape = (n_envs, rollout_steps, *obs_shape)

    obs_shm, observations = _attach_shared_array(shmem_names.observations, obs_steps_shape, np.float32)
    act_shm, actions = _attach_shared_array(shmem_names.actions, act_steps_shape, np.float32)
    rew_shm, rewards = _attach_shared_array(shmem_names.rewards, (n_envs, rollout_steps), np.float32)
    ep_shm, episode_starts = _attach_shared_array(shmem_names.episode_starts, (n_envs, rollout_steps), np.bool_)
    done_shm, dones = _attach_shared_array(shmem_names.dones, (n_envs, rollout_steps), np.bool_)
    val_shm, values = _attach_shared_array(shmem_names.values, (n_envs, rollout_steps), np.float32)
    lp_shm, log_probs = _attach_shared_array(shmem_names.log_probs, (n_envs, rollout_steps), np.float32)
    final_shm, final_observations = _attach_shared_array(shmem_names.final_observations, (n_envs, *obs_shape), np.float32)
    trunc_shm, time_limit_truncated = _attach_shared_array(
        shmem_names.time_limit_truncated, (n_envs, rollout_steps), np.bool_
    )
    term_shm, terminal_observations = _attach_shared_array(shmem_names.terminal_observations, term_obs_shape, np.float32)
    info_shm, info_scalars = _attach_shared_array(
        shmem_names.info_scalars,
        (n_envs, rollout_steps, N_ENV_INFO_KEYS),
        np.float32,
    )
    ep_ret_shm, episode_return = _attach_shared_array(shmem_names.episode_return, (n_envs, rollout_steps), np.float32)
    ep_len_shm, episode_length = _attach_shared_array(shmem_names.episode_length, (n_envs, rollout_steps), np.float32)
    ep_succ_shm, episode_success = _attach_shared_array(shmem_names.episode_success, (n_envs, rollout_steps), np.float32)

    attached = [
        obs_shm,
        act_shm,
        rew_shm,
        ep_shm,
        done_shm,
        val_shm,
        lp_shm,
        final_shm,
        trunc_shm,
        term_shm,
        info_shm,
        ep_ret_shm,
        ep_len_shm,
        ep_succ_shm,
    ]

    try:
        env = _patch_env(env_fn_wrapper.var())
        policy = build_worker_policy(env)
        policy.set_training_mode(False)
        current_obs: np.ndarray | None = None

        while True:
            try:
                cmd, data = remote.recv()
                if cmd == "set_weights":
                    policy.load_state_dict(data)
                    remote.send(None)
                elif cmd == "reset":
                    maybe_options = {"options": data[1]} if data[1] else {}
                    observation, reset_info = env.reset(seed=data[0], **maybe_options)
                    current_obs = np.asarray(observation, dtype=np.float32)
                    remote.send((current_obs, reset_info))
                elif cmd == "collect_rollout":
                    n_steps, initial_episode_start = data
                    n_steps = min(int(n_steps), rollout_steps)
                    if current_obs is None:
                        observation, _ = env.reset()
                        current_obs = np.asarray(observation, dtype=np.float32)
                    current_obs, final_reset_info = _collect_rollout_into_shmem(
                        env=env,
                        policy=policy,
                        current_obs=current_obs,
                        env_idx=env_idx,
                        n_steps=n_steps,
                        initial_episode_start=bool(initial_episode_start),
                        observations=observations,
                        actions=actions,
                        rewards=rewards,
                        episode_starts=episode_starts,
                        dones=dones,
                        values=values,
                        log_probs=log_probs,
                        final_observations=final_observations,
                        time_limit_truncated=time_limit_truncated,
                        terminal_observations=terminal_observations,
                        info_scalars=info_scalars,
                        episode_return=episode_return,
                        episode_length=episode_length,
                        episode_success=episode_success,
                    )
                    remote.send(final_reset_info)
                elif cmd == "render":
                    remote.send(env.render())
                elif cmd == "close":
                    env.close()
                    remote.close()
                    break
                elif cmd == "get_spaces":
                    remote.send((env.observation_space, env.action_space))
                elif cmd == "env_method":
                    method = env.get_wrapper_attr(data[0])
                    remote.send(method(*data[1], **data[2]))
                elif cmd == "get_attr":
                    remote.send(env.get_wrapper_attr(data))
                elif cmd == "has_attr":
                    try:
                        env.get_wrapper_attr(data)
                        remote.send(True)
                    except AttributeError:
                        remote.send(False)
                elif cmd == "set_attr":
                    env.set_wrapper_attr(data[0], data[1])
                    remote.send(None)
                elif cmd == "is_wrapped":
                    remote.send(is_wrapped(env, data))
                else:
                    raise NotImplementedError(f"`{cmd}` is not implemented in policy shmem rollout worker")
            except EOFError:
                break
            except KeyboardInterrupt:
                break
    finally:
        for shm in attached:
            shm.close()


class PolicyShmemSubprocVecEnv(VecEnv):
    """Policy rollout в воркерах; массивы rollout в shared memory, pipe — только сигнал."""

    def __init__(
        self,
        env_fns: list[Callable[[], gym.Env]],
        *,
        rollout_steps: int,
        start_method: str | None = "spawn",
    ) -> None:
        if rollout_steps <= 0:
            raise ValueError("rollout_steps must be positive")

        self.waiting = False
        self.closed = False
        self.rollout_steps = int(rollout_steps)
        n_envs = len(env_fns)
        if n_envs == 0:
            raise ValueError("PolicyShmemSubprocVecEnv requires at least one environment")

        ctx = mp.get_context(start_method or "spawn")
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)], strict=True)
        self.processes: list[mp.Process] = []
        self._shared_memories: list[shared_memory.SharedMemory] = []

        probe_env = _patch_env(env_fns[0]())
        observation_space = probe_env.observation_space
        action_space = probe_env.action_space
        probe_env.close()

        if not isinstance(observation_space, spaces.Box) or not isinstance(action_space, spaces.Box):
            raise TypeError("PolicyShmemSubprocVecEnv supports only Box observation and action spaces")

        obs_shape = observation_space.shape
        act_shape = action_space.shape
        obs_steps_shape = (n_envs, self.rollout_steps, *obs_shape)
        act_steps_shape = (n_envs, self.rollout_steps, *act_shape)
        term_obs_shape = (n_envs, self.rollout_steps, *obs_shape)

        obs_shm, self.observations = _create_shared_array(obs_steps_shape, np.float32)
        act_shm, self.actions = _create_shared_array(act_steps_shape, np.float32)
        rew_shm, self.rewards = _create_shared_array((n_envs, self.rollout_steps), np.float32)
        ep_shm, self.episode_starts = _create_shared_array((n_envs, self.rollout_steps), np.bool_)
        done_shm, self.dones = _create_shared_array((n_envs, self.rollout_steps), np.bool_)
        val_shm, self.values = _create_shared_array((n_envs, self.rollout_steps), np.float32)
        lp_shm, self.log_probs = _create_shared_array((n_envs, self.rollout_steps), np.float32)
        final_shm, self.final_observations = _create_shared_array((n_envs, *obs_shape), np.float32)
        trunc_shm, self.time_limit_truncated = _create_shared_array((n_envs, self.rollout_steps), np.bool_)
        term_shm, self.terminal_observations = _create_shared_array(term_obs_shape, np.float32)
        info_shm, self.info_scalars = _create_shared_array(
            (n_envs, self.rollout_steps, N_ENV_INFO_KEYS),
            np.float32,
        )
        ep_ret_shm, self.episode_return = _create_shared_array((n_envs, self.rollout_steps), np.float32)
        ep_len_shm, self.episode_length = _create_shared_array((n_envs, self.rollout_steps), np.float32)
        ep_succ_shm, self.episode_success = _create_shared_array((n_envs, self.rollout_steps), np.float32)

        self._shared_memories.extend(
            [
                obs_shm,
                act_shm,
                rew_shm,
                ep_shm,
                done_shm,
                val_shm,
                lp_shm,
                final_shm,
                trunc_shm,
                term_shm,
                info_shm,
                ep_ret_shm,
                ep_len_shm,
                ep_succ_shm,
            ]
        )
        self._shmem_names = _RolloutShmemNames(
            observations=obs_shm.name,
            actions=act_shm.name,
            rewards=rew_shm.name,
            episode_starts=ep_shm.name,
            dones=done_shm.name,
            values=val_shm.name,
            log_probs=lp_shm.name,
            final_observations=final_shm.name,
            time_limit_truncated=trunc_shm.name,
            terminal_observations=term_shm.name,
            info_scalars=info_shm.name,
            episode_return=ep_ret_shm.name,
            episode_length=ep_len_shm.name,
            episode_success=ep_succ_shm.name,
        )
        self._final_reset_infos: list[dict[str, Any]] = [{} for _ in range(n_envs)]

        for env_idx, (work_remote, remote, env_fn) in enumerate(
            zip(self.work_remotes, self.remotes, env_fns, strict=True)
        ):
            args = (
                work_remote,
                remote,
                CloudpickleWrapper(env_fn),
                env_idx,
                self.rollout_steps,
                self._shmem_names,
                n_envs,
                obs_shape,
                act_shape,
            )
            process = ctx.Process(target=_policy_shmem_rollout_worker, args=args, daemon=True)  # type: ignore[attr-defined]
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        super().__init__(n_envs, observation_space, action_space)

    def uses_worker_policy_rollout(self) -> bool:
        return True

    def uses_rollout_shmem(self) -> bool:
        return True

    def stacked_rollout_step(
        self,
        step_idx: int,
        n_steps: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], np.ndarray]:
        """Срез rollout по шагу из shared memory (без pickle)."""
        infos: list[dict[str, Any]] = []
        for env_idx in range(self.num_envs):
            info = self._info_dict(env_idx, step_idx)
            ep_r = self.episode_return[env_idx, step_idx]
            if not np.isnan(ep_r):
                info["episode"] = {
                    "r": float(ep_r),
                    "l": float(self.episode_length[env_idx, step_idx]),
                }
                if self.episode_success[env_idx, step_idx] > 0.5:
                    info["terminated"] = True
                else:
                    info["truncated"] = True
            infos.append(info)

        return (
            self.observations[:, step_idx],
            self.actions[:, step_idx],
            self.rewards[:, step_idx],
            self.episode_starts[:, step_idx],
            self.values[:, step_idx, np.newaxis],
            self.log_probs[:, step_idx, np.newaxis],
            infos,
            self.dones[:, step_idx],
        )

    def broadcast_policy_weights(self, policy_state_dict: dict[str, Any]) -> None:
        for remote in self.remotes:
            remote.send(("set_weights", policy_state_dict))
        for remote in self.remotes:
            remote.recv()

    def collect_policy_rollouts(
        self,
        n_steps: int,
        episode_starts: np.ndarray,
    ) -> None:
        n_steps = min(int(n_steps), self.rollout_steps)
        for remote, episode_start in zip(self.remotes, episode_starts, strict=True):
            remote.send(("collect_rollout", (n_steps, bool(episode_start))))
        for env_idx, remote in enumerate(self.remotes):
            self._final_reset_infos[env_idx] = remote.recv()

    def _info_dict(self, env_idx: int, step_idx: int) -> dict[str, Any]:
        info: dict[str, Any] = {}
        row = self.info_scalars[env_idx, step_idx]
        for key, idx in INFO_KEY_INDEX.items():
            value = row[idx]
            if not np.isnan(value):
                info[key] = float(value)
        if self.time_limit_truncated[env_idx, step_idx]:
            info["TimeLimit.truncated"] = True
            info["terminal_observation"] = self.terminal_observations[env_idx, step_idx]
        return info

    def step_async(self, actions: np.ndarray) -> None:
        raise RuntimeError("PolicyShmemSubprocVecEnv: используйте collect_policy_rollouts")

    def step_wait(self) -> VecEnvStepReturn:
        raise RuntimeError("PolicyShmemSubprocVecEnv: используйте collect_policy_rollouts")

    def reset(self) -> VecEnvObs:
        observations: list[np.ndarray] = []
        reset_infos: list[dict[str, Any]] = []
        for env_idx, remote in enumerate(self.remotes):
            remote.send(("reset", (self._seeds[env_idx], self._options[env_idx])))
        for remote in self.remotes:
            obs, reset_info = remote.recv()
            observations.append(obs)
            reset_infos.append(reset_info)
        self.reset_infos = reset_infos
        self._reset_seeds()
        self._reset_options()
        return _stack_obs(observations, self.observation_space)

    def close(self) -> None:
        if self.closed:
            return
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        for shm in self._shared_memories:
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
        self.closed = True

    def get_images(self) -> Sequence[np.ndarray | None]:
        if self.render_mode != "rgb_array":
            warnings.warn(
                f"The render mode is {self.render_mode}, but this method assumes it is `rgb_array` to obtain images."
            )
            return [None for _ in self.remotes]
        for pipe in self.remotes:
            pipe.send(("render", None))
        return [pipe.recv() for pipe in self.remotes]

    def has_attr(self, attr_name: str) -> bool:
        target_remotes = self._get_target_remotes(indices=None)
        for remote in target_remotes:
            remote.send(("has_attr", attr_name))
        return all(remote.recv() for remote in target_remotes)

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_attr", attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("set_attr", (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs,
    ) -> list[Any]:
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("env_method", (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices: VecEnvIndices = None) -> list[bool]:
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("is_wrapped", wrapper_class))
        return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(self, indices: VecEnvIndices) -> list[Any]:
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]
