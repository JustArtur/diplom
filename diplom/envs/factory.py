from __future__ import annotations

from diplom.config import EnvironmentConfig, WindConfig
from diplom.wind.factory import build_wind_interpolator

from .balloon_env import BalloonEnv


def build_env(
    env_config: EnvironmentConfig,
    wind_config: WindConfig,
    env_idx: int | None = None,
) -> BalloonEnv:
    # Возвращённая среда владеет интерполятором: вызов env.close() корректно
    # освобождает ресурсы NetCDF-файла
    return BalloonEnv(
        env_config,
        build_wind_interpolator(wind_config, env_idx=env_idx),
        env_idx=env_idx,
    )
