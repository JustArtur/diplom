# Куррикулумы обучения PPO и расписание ent_coef.

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback

from diplom.config import (
    DEFAULT_EPISODE_LENGTH_CURRICULUM_STAGES,
    EpisodeLengthCurriculumStage,
    EpisodeLengthCurriculumStageInput,
)

_LAST_STAGE_UNTIL = 10**12


def normalize_episode_length_curriculum_stages(
    stages: tuple[EpisodeLengthCurriculumStageInput, ...],
) -> tuple[EpisodeLengthCurriculumStage, ...]:
    if not stages:
        raise ValueError("episode_length_curriculum_stages must not be empty")

    normalized: list[EpisodeLengthCurriculumStage] = []
    for item in stages:
        if isinstance(item, EpisodeLengthCurriculumStage):
            normalized.append(item)
            continue
        from_timesteps, until_timesteps, max_episode_steps = item
        normalized.append(
            EpisodeLengthCurriculumStage(
                int(from_timesteps),
                None if until_timesteps is None else int(until_timesteps),
                int(max_episode_steps),
            )
        )

    if normalized[0].from_timesteps != 0:
        raise ValueError("first curriculum stage must start at from_timesteps=0")

    for idx, stage in enumerate(normalized):
        if stage.max_episode_steps <= 0:
            raise ValueError(
                f"stage {idx + 1}: max_episode_steps must be positive, got {stage.max_episode_steps}"
            )
        if stage.until_timesteps is not None and stage.until_timesteps <= stage.from_timesteps:
            raise ValueError(
                f"stage {idx + 1}: until_timesteps ({stage.until_timesteps}) "
                f"must be greater than from_timesteps ({stage.from_timesteps})"
            )
        if idx > 0:
            prev = normalized[idx - 1]
            if prev.until_timesteps is None:
                raise ValueError(f"stage {idx}: previous stage has no upper bound")
            if stage.from_timesteps != prev.until_timesteps:
                raise ValueError(
                    f"stage {idx + 1}: from_timesteps ({stage.from_timesteps}) "
                    f"must equal previous until_timesteps ({prev.until_timesteps})"
                )
        if idx < len(normalized) - 1 and stage.until_timesteps is None:
            raise ValueError(f"stage {idx + 1}: only the last stage may have until_timesteps=None")

    return tuple(normalized)


def resolve_episode_length_stage_index(
    timesteps: int,
    stages: tuple[EpisodeLengthCurriculumStage, ...],
) -> int:
    for idx, stage in enumerate(stages):
        until = stage.until_timesteps if stage.until_timesteps is not None else _LAST_STAGE_UNTIL
        if stage.from_timesteps <= timesteps < until:
            return idx
    return len(stages) - 1


def initial_episode_length_curriculum_max_steps(
    stages: tuple[EpisodeLengthCurriculumStageInput, ...],
) -> int:
    return normalize_episode_length_curriculum_stages(stages)[0].max_episode_steps


def log_episode_length_curriculum_plan(
    stages: tuple[EpisodeLengthCurriculumStageInput, ...],
    *,
    start_timesteps: int = 0,
) -> None:
    # Печатает таблицу этапов max_episode_steps в stdout при старте обучения.
    normalized = normalize_episode_length_curriculum_stages(stages)
    print("[episode_length_curriculum] max_episode_steps по global timesteps:")  # noqa: T201
    for idx, stage in enumerate(normalized):
        until_label = "∞" if stage.until_timesteps is None else f"{stage.until_timesteps:,}"
        print(  # noqa: T201
            f"  этап {idx + 1}: {stage.from_timesteps:,} ≤ timesteps < {until_label} "
            f"-> max_episode_steps={stage.max_episode_steps:,}"
        )

    active_idx = resolve_episode_length_stage_index(start_timesteps, normalized)
    active = normalized[active_idx]
    print(  # noqa: T201
        f"  сейчас: timesteps={start_timesteps:,} -> этап {active_idx + 1}, "
        f"max_episode_steps={active.max_episode_steps:,}"
    )


class TrainEpisodeLengthCurriculumCallback(BaseCallback):
    # Синхронно повышает max_episode_steps во всех средах по числу шагов обучения.

    def __init__(
        self,
        stages: tuple[EpisodeLengthCurriculumStageInput, ...] = DEFAULT_EPISODE_LENGTH_CURRICULUM_STAGES,
        *,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self._stages = normalize_episode_length_curriculum_stages(stages)
        self._active_idx = -1

    def _on_training_start(self) -> None:
        log_episode_length_curriculum_plan(
            self._stages,
            start_timesteps=self.num_timesteps,
        )
        self._apply_stage(resolve_episode_length_stage_index(self.num_timesteps, self._stages))

    def _on_step(self) -> bool:
        idx = resolve_episode_length_stage_index(self.num_timesteps, self._stages)
        if idx != self._active_idx:
            self._apply_stage(idx)
        return True

    def _apply_stage(self, idx: int) -> None:
        stage = self._stages[idx]
        self._active_idx = idx
        max_steps = stage.max_episode_steps
        self.training_env.set_attr("max_episode_steps", max_steps)
        if self.logger is not None:
            self.logger.record("curriculum/max_episode_steps", float(max_steps))
        if self.verbose:
            until = stage.until_timesteps if stage.until_timesteps is not None else _LAST_STAGE_UNTIL
            print(  # noqa: T201
                f"[episode_length_curriculum] stage {idx + 1}/{len(self._stages)} "
                f"({stage.from_timesteps:,} ≤ timesteps < {until:,}): "
                f"max_episode_steps={max_steps:,}"
            )


# Entropy coefficient schedule
class EntCoefScheduleCallback(BaseCallback):
    # ent_coef: start -> end за decay_timesteps (linear).

    def __init__(
        self,
        *,
        start: float,
        end: float,
        decay_timesteps: int,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self._start = float(start)
        self._end = float(end)
        self._decay_timesteps = max(1, int(decay_timesteps))

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / self._decay_timesteps)
        ent_coef = self._start + (self._end - self._start) * progress
        self.model.ent_coef = ent_coef
        if self.logger is not None:
            self.logger.record("train/ent_coef", float(ent_coef))
        return True
