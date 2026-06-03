from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diplom.envs.balloon_env import BalloonEnv


@dataclass(frozen=True, slots=True)
class EpisodeReplayResult:
    steps: list[dict[str, object]]
    total_reward: float
    final_obs: np.ndarray
    terminated: bool
    truncated: bool


def replay_episode_actions(
    env: BalloonEnv,
    *,
    seed: int,
    actions: list[float],
) -> EpisodeReplayResult:
    # Сбросить среду и прогнать actions заново, пересчитав физику, ветер, reward и obs.
    obs, _ = env.reset(seed=seed)
    steps: list[dict[str, object]] = []
    total_reward = 0.0
    terminated = False
    truncated = False

    for action in actions:
        obs, reward, terminated, truncated, _info = env.step(
            np.asarray([action], dtype=np.float32)
        )
        record = env.consume_step_record()
        if record:
            steps.append(record)
        total_reward += float(reward)
        if terminated or truncated:
            break

    return EpisodeReplayResult(
        steps=steps,
        total_reward=total_reward,
        final_obs=obs,
        terminated=terminated,
        truncated=truncated,
    )


def rewrite_env_current_trajectory(env: BalloonEnv, steps: list[dict[str, object]]) -> None:
    # Перезаписать JSONL текущего эпизода после сглаживания/replay.
    writer = env._steps_writer  # noqa: SLF001, точка интеграции manual mode / writer
    if writer is None:
        return
    writer.rewrite_current(steps)
    writer.flush()
