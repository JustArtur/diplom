"""Скаляры env info для TensorBoard (один shmem-блок, минимум ключей — без просадки FPS)."""

from __future__ import annotations

ENV_INFO_LOG_KEYS: tuple[str, ...] = (
    "wind_toward",
    "horizontal_distance",
    "horizontal_progress",
    "reward_progress_term",
    "reward_best_distance_term",
    "reward_distance_regression_term",
    "reward_wind_probe_best_term",
    "reward_wind_scan_term",
    "reward_ceiling_term",
    "reward_z_max_ceiling_term",
    "reward_low_altitude_stuck_term",
    "reward_climb_wind_term",
)

INFO_KEY_INDEX: dict[str, int] = {name: idx for idx, name in enumerate(ENV_INFO_LOG_KEYS)}
N_ENV_INFO_KEYS: int = len(ENV_INFO_LOG_KEYS)
