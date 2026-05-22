"""Куррикулумы обучения PPO и расписание ent_coef."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from diplom.envs.constants import (
    TARGET_REACH_RADIUS_FINAL,
    TARGET_REACH_RADIUS_INITIAL,
    TRAIN_EPISODE_LENGTH_CURRICULUM_INTERVAL,
    TRAIN_EPISODE_LENGTH_CURRICULUM_INTERVAL_GROWTH,
    TRAIN_EPISODE_LENGTH_CURRICULUM_MAX,
    TRAIN_EPISODE_LENGTH_CURRICULUM_MIN,
    TRAIN_EPISODE_LENGTH_CURRICULUM_STEP,
)
from diplom.rl.callbacks.episode_stats import EpisodeStatsCallback

_LAST_STAGE_UNTIL = 10**12


def _stage_index(timesteps: int, until_values: tuple[int, ...]) -> int:
    for idx, until in enumerate(until_values):
        if timesteps < until:
            return idx
    return len(until_values) - 1


# ── Position curriculum ──

DEFAULT_CURRICULUM_STAGES: tuple[tuple[int, np.ndarray, np.ndarray], ...] = (
    (
        5_000_000,
        np.array([8_000.0, 8_000.0, 0.0], dtype=np.float32),
        np.array([8_000.0, 8_000.0, 1_500.0], dtype=np.float32),
    ),
    (
        10_000_000,
        np.array([12_000.0, 12_000.0, 0.0], dtype=np.float32),
        np.array([12_000.0, 12_000.0, 1_500.0], dtype=np.float32),
    ),
    (
        25_000_000,
        np.array([25_000.0, 25_000.0, 0.0], dtype=np.float32),
        np.array([25_000.0, 25_000.0, 2_000.0], dtype=np.float32),
    ),
    (
        _LAST_STAGE_UNTIL,
        np.array([50_000.0, 50_000.0, 0.0], dtype=np.float32),
        np.array([50_000.0, 50_000.0, 3_000.0], dtype=np.float32),
    ),
)


@dataclass(frozen=True, slots=True)
class _PositionStage:
    until_timesteps: int
    initial_delta: np.ndarray
    target_delta: np.ndarray


class TrainPositionCurriculumCallback(BaseCallback):
    """Обновляет train_*_position_delta в средах по числу шагов обучения."""

    def __init__(
        self,
        stages: tuple[tuple[int, np.ndarray, np.ndarray], ...] = DEFAULT_CURRICULUM_STAGES,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self._stages = tuple(
            _PositionStage(int(until), np.asarray(init, dtype=np.float32), np.asarray(tgt, dtype=np.float32))
            for until, init, tgt in stages
        )
        self._active_idx = -1

    def _on_training_start(self) -> None:
        self._apply_stage(0)

    def _on_step(self) -> bool:
        idx = _stage_index(self.num_timesteps, tuple(s.until_timesteps for s in self._stages))
        if idx != self._active_idx:
            self._apply_stage(idx)
        return True

    def _apply_stage(self, idx: int) -> None:
        stage = self._stages[idx]
        self._active_idx = idx
        init = stage.initial_delta
        tgt = stage.target_delta
        self.training_env.set_attr("train_initial_position_delta", init)
        self.training_env.set_attr("train_target_position_delta", tgt)
        if self.verbose:
            print(  # noqa: T201
                f"[curriculum] stage {idx + 1}/{len(self._stages)} "
                f"(timesteps<{stage.until_timesteps}): "
                f"init_delta={init.tolist()} target_delta={tgt.tolist()}"
            )


# ── Target reach radius curriculum ──

DEFAULT_TARGET_REACH_CURRICULUM_STAGES: tuple[tuple[int, float], ...] = (
    (15_000_000, TARGET_REACH_RADIUS_INITIAL),
    (_LAST_STAGE_UNTIL, TARGET_REACH_RADIUS_FINAL),
)


@dataclass(frozen=True, slots=True)
class _TargetReachStage:
    until_timesteps: int
    target_reach_radius: float


class TrainTargetReachCurriculumCallback(BaseCallback):
    """Обновляет target_reach_radius во всех средах по global timesteps."""

    def __init__(
        self,
        stages: tuple[tuple[int, float], ...] = DEFAULT_TARGET_REACH_CURRICULUM_STAGES,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self._stages = tuple(
            _TargetReachStage(int(until), float(radius)) for until, radius in stages
        )
        self._active_idx = -1

    def _on_training_start(self) -> None:
        self._apply_stage(0)

    def _on_step(self) -> bool:
        idx = _stage_index(self.num_timesteps, tuple(s.until_timesteps for s in self._stages))
        if idx != self._active_idx:
            self._apply_stage(idx)
        return True

    def _apply_stage(self, idx: int) -> None:
        stage = self._stages[idx]
        self._active_idx = idx
        radius = stage.target_reach_radius
        self.training_env.set_attr("target_reach_radius", radius)
        if self.logger is not None:
            self.logger.record("curriculum/target_reach_radius", float(radius))
        if self.verbose:
            print(  # noqa: T201
                f"[target_reach_curriculum] stage {idx + 1}/{len(self._stages)} "
                f"(timesteps<{stage.until_timesteps}): target_reach_radius={radius:.0f} m"
            )


# ── Episode length curriculum ──

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
class _EpisodeLengthStage:
    until_timesteps: int
    max_episode_steps: int


def _normalize_episode_length_stages(
    stages: tuple[tuple[int, int], ...] | tuple[_EpisodeLengthStage, ...],
) -> tuple[_EpisodeLengthStage, ...]:
    normalized: list[_EpisodeLengthStage] = []
    for item in stages:
        if isinstance(item, _EpisodeLengthStage):
            normalized.append(item)
        else:
            until, max_steps = item
            normalized.append(_EpisodeLengthStage(int(until), int(max_steps)))
    return tuple(normalized)


def log_episode_length_curriculum_plan(
    stages: tuple[tuple[int, int], ...] | tuple[_EpisodeLengthStage, ...],
    *,
    start_timesteps: int = 0,
) -> None:
    """Печатает таблицу этапов max_episode_steps в stdout при старте обучения."""
    normalized = _normalize_episode_length_stages(stages)
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
        *,
        min_steps: int = TRAIN_EPISODE_LENGTH_CURRICULUM_MIN,
        freeze_until_success: bool = False,
        episode_stats: EpisodeStatsCallback | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self._stages = tuple(
            _EpisodeLengthStage(int(until), int(max_steps)) for until, max_steps in stages
        )
        self._min_steps = int(min_steps)
        self._freeze_until_success = bool(freeze_until_success)
        self._episode_stats = episode_stats
        self._active_idx = -1
        self._frozen = freeze_until_success

    def _on_training_start(self) -> None:
        log_episode_length_curriculum_plan(
            self._stages,
            start_timesteps=self.num_timesteps,
        )
        self._apply_stage(self._resolve_stage_index(self.num_timesteps))

    def _on_step(self) -> bool:
        if (
            self._frozen
            and self._episode_stats is not None
            and self._episode_stats.ever_succeeded
        ):
            self._frozen = False
            if self.verbose:
                print(  # noqa: T201
                    "[episode_length_curriculum] success detected — unfreezing episode length"
                )
        idx = self._resolve_stage_index(self.num_timesteps)
        if idx != self._active_idx:
            self._apply_stage(idx)
        return True

    def _resolve_stage_index(self, timesteps: int) -> int:
        if self._frozen:
            return 0
        return _stage_index(timesteps, tuple(s.until_timesteps for s in self._stages))

    def _apply_stage(self, idx: int) -> None:
        stage = self._stages[idx]
        self._active_idx = idx
        max_steps = self._min_steps if self._frozen else stage.max_episode_steps
        self.training_env.set_attr("max_episode_steps", max_steps)
        if self.logger is not None:
            self.logger.record("curriculum/max_episode_steps", float(max_steps))
        if self.verbose:
            frozen_note = " [frozen at min until success]" if self._frozen else ""
            print(  # noqa: T201
                f"[episode_length_curriculum] stage {idx + 1}/{len(self._stages)} "
                f"(timesteps<{stage.until_timesteps}): max_episode_steps={max_steps}{frozen_note}"
            )


# ── Entropy coefficient schedule ──

class EntCoefScheduleCallback(BaseCallback):
    """ent_coef: start → end за decay_timesteps (linear)."""

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
