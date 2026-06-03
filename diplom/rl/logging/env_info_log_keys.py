from __future__ import annotations

# Только метрики для диагностики навигации; остальное, в JSONL траекторий
ENV_INFO_LOG_KEYS: tuple[str, ...] = (
    "wind_toward",
    "horizontal_distance",
    "distance_to_target",
    "horizontal_progress",
    "reward_progress_term",
    "reward_goal_term",
    "reward_best_distance_term",
    "reward_distance_regression_term",
    "reward_hold_close_term",
    "reward_wind_align_term",
    "reward_wind_scan_term",
    "reward_wind_streak_term",
    "reward_z_stick_term",
)

INFO_KEY_INDEX: dict[str, int] = {name: idx for idx, name in enumerate(ENV_INFO_LOG_KEYS)}
N_ENV_INFO_KEYS: int = len(ENV_INFO_LOG_KEYS)
