"""Интерполятор ветрового поля ERA5 (u, v, w) в локальных метровых координатах."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import xarray as xr


# ──────────────────── Геометрические утилиты ────────────────────


def meters_per_deg_lat(latitude_deg: float) -> float:
    """Приблизительное число метров в одном градусе широты (WGS84).

    Ряд Фурье по геодезической модели WGS-84:
        M(φ) ≈ 111132.92 − 559.82·cos(2φ) + 1.175·cos(4φ) − 0.0023·cos(6φ)
    """
    lat_rad = math.radians(latitude_deg)
    return 111132.92 - 559.82 * math.cos(2 * lat_rad) + 1.175 * math.cos(4 * lat_rad) - 0.0023 * math.cos(6 * lat_rad)


def meters_per_deg_lon(latitude_deg: float) -> float:
    """Приблизительное число метров в одном градусе долготы (WGS84).

    Ряд Фурье по геодезической модели WGS-84:
        N(φ) ≈ 111412.84·cos(φ) − 93.5·cos(3φ) + 0.118·cos(5φ)
    """
    lat_rad = math.radians(latitude_deg)
    return 111412.84 * math.cos(lat_rad) - 93.5 * math.cos(3 * lat_rad) + 0.118 * math.cos(5 * lat_rad)


def altitude_to_pressure_hpa(height_m: np.ndarray) -> np.ndarray:
    """Давление (гПа) по высоте через барометрическую формулу стандартной атмосферы.

    Барометрическая формула ISA(международная стандартная атмосфера) (ниже тропопаузы ≈ 11 км):
        p(h) = p₀ · (1 − L·h / T₀) ^ (g·M / (R·L))

    где p₀ = 1013.25 гПа, T₀ = 288.15 K, L = 0.0065 K/м,
        g = 9.80665 м/с², M = 0.0289644 кг/моль, R = 8.31447 Дж/(моль·К).
    """
    p0 = 1013.25  # гПа
    T0 = 288.15  # K
    g = 9.80665  # м/с²
    L = 0.0065  # K/м (температурный градиент тропосферы)
    R = 8.31447  # Дж/(моль·К) — универсальная газовая постоянная
    M = 0.0289644  # кг/моль — молярная масса сухого воздуха

    height = np.asarray(height_m, dtype=float)
    # показатель степени: g·M / (R·L) ≈ 5.2559
    exponent = (g * M) / (R * L)
    return p0 * np.power(1 - (L * height) / T0, exponent)


# ──────────────────── Конвертация вертикальной скорости ────────────────────

def omega_to_w_mps(omega_pa_s: np.ndarray, pressure_hpa: np.ndarray, temperature_k: np.ndarray, ) -> np.ndarray:
    """Конвертация давленческой скорости omega (Па/с) → вертикальная скорость w (м/с).

    Из уравнения гидростатики:  dp = −ρ·g·dz
    Отсюда:  ω = dp/dt = −ρ·g·(dz/dt) = −ρ·g·w
    Значит:  w = −ω / (ρ·g)

    Плотность по уравнению состояния идеального газа:  ρ = p / (Rₐ·T)
    Подставляя:
        w = −ω · Rₐ · T / (p · g)

    где Rₐ = 287.058 Дж/(кг·К) — удельная газовая постоянная сухого воздуха,
        g = 9.80665 м/с².

    Положительное w = восходящее движение.
    """

    r_dry = 287.058  # газовая постоянная сухого воздуха (Дж/(кг·К))
    g = 9.80665  # ускорение свободного падения (м/с²)

    p_pa = np.asarray(pressure_hpa, dtype=float) * 100.0  # гПа → Па
    safe_p = np.where(p_pa > 1.0, p_pa, 1.0)  # защита от деления на 0
    omega = np.asarray(omega_pa_s, dtype=float)
    temp = np.asarray(temperature_k, dtype=float)
    # w = −ω · Rₐ · T / (p · g)
    return -omega * r_dry * temp / (safe_p * g)


# ──────────────────── Имена переменных датасета ────────────────────

WIND_U_NAME = "u"
WIND_V_NAME = "v"
WIND_W_NAME = "w"
WIND_T_NAME = "t"
LON_NAME = "longitude"
LAT_NAME = "latitude"
LEVEL_NAME = "pressure_level"
TIME_NAME = "valid_time"


# ──────────────────── Интерполятор ────────────────────


@dataclass
class WindInterpolator:
    """Интерполятор ERA5: (x_м, y_м, z_м, time) → (u, v, w) м/с."""

    ds: xr.Dataset
    origin_lat: float
    origin_lon: float

    def __post_init__(self) -> None:
        lat_coord = self.ds[LAT_NAME]
        lon_coord = self.ds[LON_NAME]
        time_coord = self.ds[TIME_NAME]
        self._lat_min = float(lat_coord.min())
        self._lat_max = float(lat_coord.max())
        self._lon_min = float(lon_coord.min())
        self._lon_max = float(lon_coord.max())
        self._time_min = np.datetime64(time_coord.min().values)
        self._time_max = np.datetime64(time_coord.max().values)

    # ──────────────────── Фабрика ────────────────────

    @classmethod
    def from_file(cls, path: Path, origin_lat: float, origin_lon: float) -> WindInterpolator:
        """Открыть датасет и создать интерполятор с фиксированными именами."""
        return cls(ds=xr.open_dataset(path), origin_lat=origin_lat, origin_lon=origin_lon)

    # ──────────────────── Преобразование координат ────────────────────

    def _xy_to_latlon(self, x_m: np.ndarray, y_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Локальные метры (x, y) → (lat, lon) относительно origin."""
        m_per_lat = meters_per_deg_lat(self.origin_lat)
        m_per_lon = meters_per_deg_lon(self.origin_lat)
        lat = np.clip(self.origin_lat + y_m / m_per_lat, self._lat_min, self._lat_max)
        lon = np.clip(self.origin_lon + x_m / m_per_lon, self._lon_min, self._lon_max)
        return lat, lon

    def _z_to_pressure(self, z_m: np.ndarray) -> np.ndarray:
        """Высота (м) → давление (гПа) с кламп к диапазону датасета."""
        level_coord = self.ds[LEVEL_NAME]
        p = altitude_to_pressure_hpa(z_m)
        p_min, p_max = float(level_coord.min()), float(level_coord.max())
        return np.clip(p, min(p_min, p_max), max(p_min, p_max))

    # ──────────────────── Интерполяция ────────────────────

    def _interp_at(self, lat: np.ndarray, lon: np.ndarray, level: np.ndarray, t: np.ndarray, ) -> xr.Dataset:
        """4D-линейная интерполяция ERA5 в заданных точках. xarray сама делает всю магию интерполяции."""
        return self.ds[[WIND_U_NAME, WIND_V_NAME, WIND_W_NAME, WIND_T_NAME]].interp(
            {
                LON_NAME: ("points", lon),
                LAT_NAME: ("points", lat),
                LEVEL_NAME: ("points", level),
                TIME_NAME: ("points", t),
            },
            method="linear",
        )

    # ──────────────────── Публичный API ────────────────────

    def vector_at(self, x: float, y: float, z: float, time: np.datetime64) -> Tuple[
        float, float, float, float]:
        """Вектор ветра (u, v, w) м/с в одной точке. w > 0 = вверх."""
        lat, lon = self._xy_to_latlon(np.array([x]), np.array([y]))
        level = self._z_to_pressure(np.array([z]))
        t = np.array([time])

        pt = self._interp_at(lat, lon, level, t)
        u = pt[WIND_U_NAME].values[0]
        v = pt[WIND_V_NAME].values[0]
        w = omega_to_w_mps(pt[WIND_W_NAME].values, level, pt[WIND_T_NAME].values)[0]
        t = pt[WIND_T_NAME].values[0]
        return u, v, w, t

    def batch_vector_at(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, time: np.ndarray) -> np.ndarray:
        """Векторы ветра (u, v, w) м/с для батча точек. Shape: (n, 3)."""
        lat, lon = self._xy_to_latlon(x, y)
        level = self._z_to_pressure(z)

        pt = self._interp_at(lat, lon, level, time)
        u = pt[WIND_U_NAME].values
        v = pt[WIND_V_NAME].values
        w = omega_to_w_mps(pt[WIND_W_NAME].values, level, pt[WIND_T_NAME].values)

        return np.stack([u, v, w], axis=-1)

    def close(self) -> None:
        self.ds.close()
