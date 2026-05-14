"""Фабрики для создания объектов, связанных с ветровым полем."""

from __future__ import annotations

from diplom.config import WindConfig

from .interp import WindInterpolator


def build_wind_interpolator(config: WindConfig) -> WindInterpolator:
    """Создать `WindInterpolator` из конфигурации ветра."""
    return WindInterpolator.from_file(
        path=config.path,
        origin_lat=config.origin_lat,
        origin_lon=config.origin_lon,
    )
