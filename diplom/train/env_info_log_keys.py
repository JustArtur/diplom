"""Скаляры env info для TensorBoard (один shmem-блок, минимум ключей — без просадки FPS)."""

from __future__ import annotations

# Только метрики для диагностики навигации; остальное — в JSONL траекторий.
ENV_INFO_LOG_KEYS: tuple[str, ...] = (
    "wind_toward",
    "reward_wind_align_term",
    "reward_wind_streak_term",
    "reward_wind_adverse_streak_term",
    "horizontal_distance",
    "distance_to_target",
    "horizontal_progress",
    "reward_boundary_term",
)

INFO_KEY_INDEX: dict[str, int] = {name: idx for idx, name in enumerate(ENV_INFO_LOG_KEYS)}
N_ENV_INFO_KEYS: int = len(ENV_INFO_LOG_KEYS)
