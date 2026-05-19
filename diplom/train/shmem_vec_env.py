"""SubprocVecEnv с shared memory для obs/reward/done (без pickle на каждый step)."""

from __future__ import annotations

import multiprocessing as mp
import warnings
from collections.abc import Callable, Sequence
from multiprocessing import shared_memory
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)
from stable_baselines3.common.vec_env.patch_gym import _patch_env
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


def _flat_action(action: np.ndarray) -> np.ndarray:
    return np.asarray(action, dtype=np.float32).reshape(-1)


def _shmem_worker(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
    env_idx: int,
    obs_shm_name: str,
    rew_shm_name: str,
    done_shm_name: str,
    act_shm_name: str,
    obs_shape: tuple[int, ...],
    act_shape: tuple[int, ...],
) -> None:
    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()
    obs_shm, obs_buf = _attach_shared_array(obs_shm_name, obs_shape, np.float32)
    rew_shm, rew_buf = _attach_shared_array(rew_shm_name, (obs_shape[0],), np.float32)
    done_shm, done_buf = _attach_shared_array(done_shm_name, (obs_shape[0],), np.bool_)
    act_shm, act_buf = _attach_shared_array(act_shm_name, act_shape, np.float32)

    try:
        env = _patch_env(env_fn_wrapper.var())
        reset_info: dict[str, Any] = {}

        while True:
            try:
                cmd, data = remote.recv()
                if cmd == "step":
                    action = _flat_action(act_buf[env_idx])
                    observation, reward, terminated, truncated, info = env.step(action)
                    done = bool(terminated or truncated)
                    info["TimeLimit.truncated"] = bool(truncated and not terminated)
                    if done:
                        info["terminal_observation"] = observation
                        observation, reset_info = env.reset()
                    obs_buf[env_idx] = np.asarray(observation, dtype=np.float32)
                    rew_buf[env_idx] = float(reward)
                    done_buf[env_idx] = done
                    remote.send((info, reset_info))
                elif cmd == "reset":
                    maybe_options = {"options": data[1]} if data[1] else {}
                    observation, reset_info = env.reset(seed=data[0], **maybe_options)
                    obs_buf[env_idx] = np.asarray(observation, dtype=np.float32)
                    remote.send(reset_info)
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
                    remote.send(setattr(env, data[0], data[1]))  # type: ignore[func-returns-value]
                elif cmd == "is_wrapped":
                    remote.send(is_wrapped(env, data))
                else:
                    raise NotImplementedError(f"`{cmd}` is not implemented in the shmem worker")
            except EOFError:
                break
            except KeyboardInterrupt:
                break
    finally:
        obs_shm.close()
        rew_shm.close()
        done_shm.close()
        act_shm.close()


class ShmemSubprocVecEnv(VecEnv):
    """Векторная среда: шаги через shared memory, pipe только для лёгких команд."""

    def __init__(
        self,
        env_fns: list[Callable[[], gym.Env]],
        start_method: str | None = "spawn",
    ) -> None:
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)
        if n_envs == 0:
            raise ValueError("ShmemSubprocVecEnv requires at least one environment")

        ctx = mp.get_context(start_method or "spawn")

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)], strict=True)
        self.processes: list[mp.Process] = []
        self._shared_memories: list[shared_memory.SharedMemory] = []

        # Временный env для форм проб (только в родителе, до spawn).
        probe_env = _patch_env(env_fns[0]())
        observation_space = probe_env.observation_space
        action_space = probe_env.action_space
        probe_env.close()

        if not isinstance(observation_space, spaces.Box) or not isinstance(action_space, spaces.Box):
            raise TypeError("ShmemSubprocVecEnv supports only Box observation and action spaces")

        obs_shape = (n_envs, *observation_space.shape)
        act_shape = (n_envs, *action_space.shape)

        obs_shm, self.observations = _create_shared_array(obs_shape, np.float32)
        rew_shm, self.rewards = _create_shared_array((n_envs,), np.float32)
        done_shm, self.dones = _create_shared_array((n_envs,), np.bool_)
        act_shm, self.actions = _create_shared_array(act_shape, np.float32)
        self._shared_memories.extend([obs_shm, rew_shm, done_shm, act_shm])

        obs_shm_name = obs_shm.name
        rew_shm_name = rew_shm.name
        done_shm_name = done_shm.name
        act_shm_name = act_shm.name
        per_env_obs_shape = (n_envs, *observation_space.shape)
        per_env_act_shape = (n_envs, *action_space.shape)

        for env_idx, (work_remote, remote, env_fn) in enumerate(
            zip(self.work_remotes, self.remotes, env_fns, strict=True)
        ):
            args = (
                work_remote,
                remote,
                CloudpickleWrapper(env_fn),
                env_idx,
                obs_shm_name,
                rew_shm_name,
                done_shm_name,
                act_shm_name,
                per_env_obs_shape,
                per_env_act_shape,
            )
            process = ctx.Process(target=_shmem_worker, args=args, daemon=True)  # type: ignore[attr-defined]
            process.start()
            self.processes.append(process)
            work_remote.close()

        super().__init__(n_envs, observation_space, action_space)

    def step_async(self, actions: np.ndarray) -> None:
        self.actions[:] = np.asarray(actions, dtype=np.float32)
        for remote in self.remotes:
            remote.send(("step", None))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        infos: list[dict[str, Any]] = []
        reset_infos: list[dict[str, Any]] = []
        for remote in self.remotes:
            info, reset_info = remote.recv()
            infos.append(info)
            reset_infos.append(reset_info)
        self.waiting = False
        self.reset_infos = reset_infos
        return (
            self.observations.copy(),
            self.rewards.copy(),
            self.dones.copy(),
            infos,
        )

    def reset(self) -> VecEnvObs:
        for env_idx, remote in enumerate(self.remotes):
            remote.send(("reset", (self._seeds[env_idx], self._options[env_idx])))
        reset_infos = [remote.recv() for remote in self.remotes]
        self.reset_infos = list(reset_infos)
        self._reset_seeds()
        self._reset_options()
        return self.observations.copy()

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
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
