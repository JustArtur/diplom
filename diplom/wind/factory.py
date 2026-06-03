from __future__ import annotations

from diplom.config import WindConfig

from .interp import WindInterpolator


def build_wind_interpolator(config: WindConfig, env_idx: int | None = None) -> WindInterpolator:
    return WindInterpolator.from_file(
        path=config.path,
        env_idx=env_idx,
        origin_lat=config.origin_lat,
        origin_lon=config.origin_lon,
    )
