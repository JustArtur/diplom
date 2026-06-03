# PPO: policy + env.step в subprocess, rollout через shared memory.

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from diplom.rl.vec_env.policy_shmem_rollout import PolicyShmemSubprocVecEnv


class WorkerRolloutPPO(PPO):
    # PPO для PolicyShmemSubprocVecEnv (гибрид worker policy + rollout shmem).

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: Any,
        n_rollout_steps: int,
    ) -> bool:
        if not isinstance(env, PolicyShmemSubprocVecEnv):
            return super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)

        if self.use_sde:
            raise NotImplementedError("use_sde не поддерживается с worker policy rollout")

        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)
        rollout_buffer.reset()
        callback.on_rollout_start()

        policy_state = {key: tensor.detach().cpu() for key, tensor in self.policy.state_dict().items()}
        env.broadcast_policy_weights(policy_state)

        episode_starts_in = np.asarray(self._last_episode_starts, dtype=bool)
        env.collect_policy_rollouts(n_rollout_steps, episode_starts_in)

        for step_idx in range(n_rollout_steps):
            obs, actions, rewards, episode_starts, values, log_probs, infos, dones = env.stacked_rollout_step(
                step_idx, n_rollout_steps
            )
            rewards = _bootstrap_truncated_rewards(self, rewards, dones, infos)

            self.num_timesteps += env.num_envs
            callback.update_locals(
                {
                    "self": self,
                    "env": env,
                    "callback": callback,
                    "rollout_buffer": rollout_buffer,
                    "n_rollout_steps": n_rollout_steps,
                    "n_steps": step_idx,
                    "actions": actions,
                    "values": values,
                    "log_probs": log_probs,
                    "rewards": rewards,
                    "dones": dones,
                    "infos": infos,
                    "new_obs": None,
                }
            )
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            rollout_buffer.add(
                obs,
                actions,
                rewards,
                episode_starts,
                th.as_tensor(values, dtype=th.float32, device=self.device).reshape(env.num_envs),
                th.as_tensor(log_probs, dtype=th.float32, device=self.device).reshape(env.num_envs),
            )

        new_obs = env.final_observations.copy()
        dones = env.dones[:, n_rollout_steps - 1].copy()

        with th.no_grad():
            last_values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=dones)
        self._last_obs = new_obs
        self._last_episode_starts = dones

        callback.update_locals(locals())
        callback.on_rollout_end()
        return True


def _bootstrap_truncated_rewards(
    model: PPO,
    rewards: np.ndarray,
    dones: np.ndarray,
    infos: list[dict],
) -> np.ndarray:
    adjusted = rewards.copy()
    for env_idx, done in enumerate(dones):
        if (
            done
            and infos[env_idx].get("terminal_observation") is not None
            and infos[env_idx].get("TimeLimit.truncated", False)
        ):
            terminal_obs = model.policy.obs_to_tensor(infos[env_idx]["terminal_observation"])[0]
            with th.no_grad():
                terminal_value = model.policy.predict_values(terminal_obs)[0]
            adjusted[env_idx] += float(model.gamma * terminal_value)
    return adjusted
