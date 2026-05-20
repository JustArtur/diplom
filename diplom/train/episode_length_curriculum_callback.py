"""Куррикулум: постепенное увеличение max_episode_steps по global timesteps."""

from __future__ import annotations

from dataclasses import dataclass

from stable_baselines3.common.callbacks import BaseCallback

from diplom.envs.constants import (
    TRAIN_EPISODE_LENGTH_CURRICULUM_INTERVAL,
    TRAIN_EPISODE_LENGTH_CURRICULUM_INTERVAL_GROWTH,
    TRAIN_EPISODE_LENGTH_CURRICULUM_MAX,
    TRAIN_EPISODE_LENGTH_CURRICULUM_MIN,
    TRAIN_EPISODE_LENGTH_CURRICULUM_STEP,
)

_LAST_STAGE_UNTIL = 10**12


def build_episode_length_curriculum_stages(
    *,
    min_steps: int = TRAIN_EPISODE_LENGTH_CURRICULUM_MIN,
    max_steps: int = TRAIN_EPISODE_LENGTH_CURRICULUM_MAX,
    step: int = TRAIN_EPISODE_LENGTH_CURRICULUM_STEP,
    interval: int = TRAIN_EPISODE_LENGTH_CURRICULUM_INTERVAL,
    interval_growth: int = TRAIN_EPISODE_LENGTH_CURRICULUM_INTERVAL_GROWTH,
) -> tuple[tuple[int, int], ...]:
    """(until_timesteps, max_episode_steps) — этап активен, пока num_timesteps < until."""
    if min_steps > max_steps:
        raise ValueError(f"min_steps ({min_steps}) must be <= max_steps ({max_steps})")
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")
    if interval <= 0:
        raise ValueError(f"interval must be positive, got {interval}")
    if interval_growth < 0:
        raise ValueError(f"interval_growth must be non-negative, got {interval_growth}")

    limits: list[int] = []
    current = min_steps
    while True:
        limits.append(current)
        if current >= max_steps:
            break
        current += step

    stages: list[tuple[int, int]] = []
    until = 0
    for idx, limit in enumerate(limits):
        if idx < len(limits) - 1:
            stage_interval = interval + idx * interval_growth
            until += stage_interval
            stages.append((until, limit))
        else:
            stages.append((_LAST_STAGE_UNTIL, limit))
    return tuple(stages)


DEFAULT_EPISODE_LENGTH_CURRICULUM_STAGES = build_episode_length_curriculum_stages()


@dataclass(frozen=True, slots=True)
class _Stage:
    until_timesteps: int
    max_episode_steps: int


def _normalize_stages(
    stages: tuple[tuple[int, int], ...] | tuple[_Stage, ...],
) -> tuple[_Stage, ...]:
    normalized: list[_Stage] = []
    for item in stages:
        if isinstance(item, _Stage):
            normalized.append(item)
        else:
            until, max_steps = item
            normalized.append(_Stage(int(until), int(max_steps)))
    return tuple(normalized)


def log_episode_length_curriculum_plan(
    stages: tuple[tuple[int, int], ...] | tuple[_Stage, ...],
    *,
    start_timesteps: int = 0,
) -> None:
    """Печатает таблицу этапов max_episode_steps в stdout при старте обучения."""
    normalized = _normalize_stages(stages)
    if not normalized:
        return

    min_steps = normalized[0].max_episode_steps
    max_steps = normalized[-1].max_episode_steps
    ep_step = (
        normalized[1].max_episode_steps - normalized[0].max_episode_steps
        if len(normalized) > 1
        else 0
    )

    print("[episode_length_curriculum] max_episode_steps по global timesteps:")  # noqa: T201
    if ep_step > 0 and len(normalized) > 1:
        first_duration = normalized[0].until_timesteps
        second_duration = (
            normalized[1].until_timesteps - normalized[0].until_timesteps
            if len(normalized) > 1
            else 0
        )
        growth_hint = ""
        if second_duration > first_duration:
            growth_hint = f", далее +{second_duration - first_duration:,} timesteps на этап"
        print(  # noqa: T201
            f"  правило: лимит эпизода от {min_steps:,} до {max_steps:,} с шагом +{ep_step:,}; "
            f"длительность этапа: {first_duration:,}{growth_hint}"
        )
    for idx, stage in enumerate(normalized):
        from_ts = 0 if idx == 0 else normalized[idx - 1].until_timesteps
        if stage.until_timesteps >= _LAST_STAGE_UNTIL:
            print(f"  этап {idx + 1}: timesteps ≥ {from_ts:,} → {stage.max_episode_steps:,}")  # noqa: T201
        else:
            stage_duration = stage.until_timesteps - from_ts
            print(  # noqa: T201
                f"  этап {idx + 1}: {from_ts:,} ≤ timesteps < {stage.until_timesteps:,} "
                f"({stage_duration:,} timesteps) → {stage.max_episode_steps:,}"
            )

    active_idx = 0
    for idx, stage in enumerate(normalized):
        if start_timesteps < stage.until_timesteps:
            active_idx = idx
            break
        active_idx = idx
    active = normalized[active_idx]
    print(  # noqa: T201
        f"  сейчас: timesteps={start_timesteps:,} → этап {active_idx + 1}, "
        f"max_episode_steps={active.max_episode_steps:,}"
    )


class TrainEpisodeLengthCurriculumCallback(BaseCallback):
    """Синхронно повышает max_episode_steps во всех средах по числу шагов обучения."""

    def __init__(
        self,
        stages: tuple[tuple[int, int], ...] = DEFAULT_EPISODE_LENGTH_CURRICULUM_STAGES,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self._stages = tuple(
            _Stage(int(until), int(max_steps)) for until, max_steps in stages
        )
        self._active_idx = -1

    def _on_training_start(self) -> None:
        log_episode_length_curriculum_plan(
            self._stages,
            start_timesteps=self.num_timesteps,
        )
        self._apply_stage(self._stage_index(self.num_timesteps))

    def _on_step(self) -> bool:
        idx = self._stage_index(self.num_timesteps)
        if idx != self._active_idx:
            self._apply_stage(idx)
        return True

    def _stage_index(self, timesteps: int) -> int:
        for idx, stage in enumerate(self._stages):
            if timesteps < stage.until_timesteps:
                return idx
        return len(self._stages) - 1

    def _apply_stage(self, idx: int) -> None:
        stage = self._stages[idx]
        self._active_idx = idx
        max_steps = stage.max_episode_steps
        self.training_env.set_attr("max_episode_steps", max_steps)
        if self.logger is not None:
            self.logger.record("curriculum/max_episode_steps", float(max_steps))
        if self.verbose:
            print(  # noqa: T201
                f"[episode_length_curriculum] stage {idx + 1}/{len(self._stages)} "
                f"(timesteps<{stage.until_timesteps}): max_episode_steps={max_steps}"
            )
