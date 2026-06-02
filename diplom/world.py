"""Геометрия симуляционного мира, вычисляемая из ERA5-датасета."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from diplom.config import BalloonConfig
from diplom.geo import meters_per_deg_lat, meters_per_deg_lon, pressure_hpa_to_altitude_m
from diplom.shared_constants import MAX_HEIGHT, MIN_HEIGHT

DEFAULT_TARGET_ALTITUDE = 18_000.0
# Фиксированная стартовая высота аэростата (м AMSL).
DEFAULT_START_ALTITUDE = 100.0


@dataclass(frozen=True, slots=True)
class WorldBounds:
    """Прямоугольные границы мира в локальных метрах (X/Y) и по высоте AMSL (Z)."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def z_span(self) -> float:
        return self.z_max - self.z_min

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.x_min + self.x_max) / 2.0,
            (self.y_min + self.y_max) / 2.0,
        )


def world_bounds_from_axes(
        latitude_axis_deg: np.ndarray,
        longitude_axis_deg: np.ndarray,
        *,
        origin_lat: float,
        origin_lon: float,
        pressure_axis_hpa: np.ndarray | None = None,
) -> WorldBounds:
    """Перевести диапазон широт и долгот в прямоугольник локальных метров.

    Если заданы уровни давления ERA5, вертикальный диапазон Z совпадает с их эквивалентом по ISA;
    иначе используются глобальные константы MIN_HEIGHT / MAX_HEIGHT.
    """
    lat_min = float(np.min(latitude_axis_deg))
    lat_max = float(np.max(latitude_axis_deg))
    lon_min = float(np.min(longitude_axis_deg))
    lon_max = float(np.max(longitude_axis_deg))

    m_per_lat = meters_per_deg_lat(origin_lat)
    m_per_lon = meters_per_deg_lon(origin_lat)

    x_min = (lon_min - origin_lon) * m_per_lon
    x_max = (lon_max - origin_lon) * m_per_lon
    y_min = (lat_min - origin_lat) * m_per_lat
    y_max = (lat_max - origin_lat) * m_per_lat

    if pressure_axis_hpa is not None:
        z_amsl = pressure_hpa_to_altitude_m(np.asarray(pressure_axis_hpa, dtype=np.float64))
        z_lo = float(np.min(z_amsl))
        z_hi = float(np.max(z_amsl))
        z_min = float(min(z_lo, z_hi))
        z_max = float(max(z_lo, z_hi))
    else:
        z_min = MIN_HEIGHT
        z_max = MAX_HEIGHT

    return WorldBounds(
        x_min=float(min(x_min, x_max)),
        x_max=float(max(x_min, x_max)),
        y_min=float(min(y_min, y_max)),
        y_max=float(max(y_min, y_max)),
        z_min=z_min,
        z_max=z_max,
    )


def log_world_bounds(
        bounds: WorldBounds,
        *,
        origin_lat: float,
        origin_lon: float,
        wind_path: Path | str,
        prefix: str = "[world]",
        # flush: bool = True,
) -> None:

    print(f"{prefix} Размеры мира из датасета · NetCDF `{Path(wind_path)}`: \n"
           f"X [{bounds.x_min:.1f} … {bounds.x_max:.1f}] м \n"
           f"Y [{bounds.y_min:.1f} … {bounds.y_max:.1f}] м \n"
           f"Z [{bounds.z_min:.1f} … {bounds.z_max:.1f}] м \n"
           f"origin lat={origin_lat:.5f}°, lon={origin_lon:.5f}°")


def default_initial_position(bounds: WorldBounds) -> np.ndarray:
    """Базовая стартовая точка в центре мира на фиксированной высоте."""
    center_x, center_y = bounds.center
    z = float(np.clip(DEFAULT_START_ALTITUDE, bounds.z_min, bounds.z_max))
    return np.array([center_x, center_y, z], dtype=np.float32)


def default_target_position(bounds: WorldBounds) -> np.ndarray:
    """Базовая целевая точка в центре мира на рабочей высоте."""
    center_x, center_y = bounds.center
    z = float(np.clip(DEFAULT_TARGET_ALTITUDE, bounds.z_min, bounds.z_max))
    return np.array([center_x, center_y, z], dtype=np.float32)


def resolve_sim_time(
    sim_time: np.datetime64 | None,
    *,
    time_min: np.datetime64,
) -> np.datetime64:
    """Подставить начало временной оси ERA5, если момент старта не задан явно."""
    return time_min if sim_time is None else sim_time


def resolve_balloon_config(balloon: BalloonConfig, bounds: WorldBounds) -> BalloonConfig:
    """Подставить координаты по умолчанию, если они не заданы явно."""
    initial_position = (
        np.array(balloon.initial_position, dtype=np.float32)
        if balloon.initial_position is not None
        else default_initial_position(bounds)
    )
    target_position = (
        np.array(balloon.target_position, dtype=np.float32)
        if balloon.target_position is not None
        else default_target_position(bounds)
    )
    return replace(
        balloon,
        initial_position=initial_position,
        target_position=target_position,
    )
