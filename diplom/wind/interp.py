"""Интерполятор ветрового поля ERA5 (u, v, w) в локальных метровых координатах."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator


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

    height = np.asarray(height_m, dtype=np.float32)
    # Формула ISA корректна только ниже ~44.3 км; выше этого порога база степени становится невалидной.
    max_height = (T0 / L) - 1e-6
    height = np.clip(height, 0.0, max_height)
    # показатель степени: g·M / (R·L) ≈ 5.2559
    exponent = (g * M) / (R * L)
    return np.asarray(p0 * np.power(1 - (L * height) / T0, exponent), dtype=np.float32)


# ──────────────────── Конвертация вертикальной скорости ────────────────────

def omega_to_w_mps(omega_pa_s: np.ndarray, pressure_hpa: np.ndarray, temperature_k: np.ndarray, ) -> np.ndarray:
    """Конвертация давленческой скорости omega (Па/с) → вертикальная скорость w (м/с).

    Из уравнения гидростатики:  dp = −ρ·g·dz
    Отсюда:  ω = dp/dt = −ρ·g·(dz/dt) = −ρ·g·w
    Значит:  w = −ω / (ρ·g)

    Плотность по уравнению состояния идеального газа:  p = p / (Rₐ·T)
    Подставляя:
        w = −ω · Rₐ · T / (p · g)

    где Rₐ = 287.058 Дж/(кг·К) — удельная газовая постоянная сухого воздуха,
        g = 9.80665 м/с².

    Положительное w = восходящее движение.
    """

    r_dry = 287.058  # газовая постоянная сухого воздуха (Дж/(кг·К))
    g = 9.80665  # ускорение свободного падения (м/с²)

    p_pa = np.asarray(pressure_hpa, dtype=np.float32) * np.float32(100.0)  # гПа → Па
    safe_p = np.where(p_pa > 1.0, p_pa, np.float32(1.0))  # защита от деления на 0
    omega = np.nan_to_num(np.asarray(omega_pa_s, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    temp = np.nan_to_num(np.asarray(temperature_k, dtype=np.float32), nan=288.15, posinf=288.15, neginf=288.15)
    # w = −ω · Rₐ · T / (p · g)
    return np.asarray(
        np.nan_to_num(-omega * r_dry * temp / (safe_p * g), nan=0.0, posinf=0.0, neginf=0.0),
        dtype=np.float32,
    )


# ──────────────────── Имена переменных датасета (внутренние) ────────────────────
# Эти строки — детали конкретного ERA5 NetCDF-файла, а не публичный контракт модуля.

_WIND_U_NAME = "u"
_WIND_V_NAME = "v"
_WIND_W_NAME = "w"
_WIND_T_NAME = "t"
_LON_NAME = "longitude"
_LAT_NAME = "latitude"
_LEVEL_NAME = "pressure_level"
_TIME_NAME = "valid_time"


# ──────────────────── Результат интерполяции ────────────────────

@dataclass(frozen=True)
class WindSample:
    """Результат интерполяции ветра в одной точке пространства-времени."""

    u: float            # западно-восточная компонента (м/с)
    v: float            # южно-северная компонента (м/с)
    w: float            # вертикальная компонента (м/с, вверх — положительно)
    temperature: float  # температура воздуха (K)
    pressure: float     # давление (гПа)


# ──────────────────── Интерполятор ────────────────────


@dataclass
class WindInterpolator:
    """Интерполятор ERA5: (x_м, y_м, z_м, time) → (u, v, w) м/с.

    При инициализации все данные ERA5 конвертируются в numpy-массивы и
    строятся scipy.RegularGridInterpolator — это происходит один раз,
    зато каждый вызов vector_at() работает в ~50–100× быстрее, чем
    xarray.interp().
    """

    ds: xr.Dataset
    origin_lat: float
    origin_lon: float

    # scipy-интерполяторы — строятся в __post_init__, не передаются извне
    _interp_u: RegularGridInterpolator = field(init=False, repr=False)
    _interp_v: RegularGridInterpolator = field(init=False, repr=False)
    _interp_w: RegularGridInterpolator = field(init=False, repr=False)
    _interp_t: RegularGridInterpolator = field(init=False, repr=False)
    _lat_min: float = field(init=False, repr=False)
    _lat_max: float = field(init=False, repr=False)
    _lon_min: float = field(init=False, repr=False)
    _lon_max: float = field(init=False, repr=False)
    _p_min: float = field(init=False, repr=False)
    _p_max: float = field(init=False, repr=False)
    _time_min: np.datetime64 = field(init=False, repr=False)
    _time_max: np.datetime64 = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._build_scipy_interpolators()

    def _build_scipy_interpolators(self) -> None:
        """Конвертировать ERA5 xr.Dataset в scipy RegularGridInterpolator.

        RegularGridInterpolator требует строго монотонно возрастающих осей,
        поэтому каждую ось сортируем и при необходимости переставляем данные.
        """
        ds = self.ds

        # ── Извлечь оси ──────────────────────────────────────────────────────
        times_raw = ds[_TIME_NAME].values.astype("datetime64[ns]").astype(np.float64)
        levels_raw = ds[_LEVEL_NAME].values.astype(np.float64)
        lats_raw = ds[_LAT_NAME].values.astype(np.float64)
        lons_raw = ds[_LON_NAME].values.astype(np.float64)

        # ── Индексы сортировки для монотонности ───────────────────────────────
        t_idx = np.argsort(times_raw)
        l_idx = np.argsort(levels_raw)
        lat_idx = np.argsort(lats_raw)
        lon_idx = np.argsort(lons_raw)

        times = times_raw[t_idx]
        levels = levels_raw[l_idx]
        lats = lats_raw[lat_idx]
        lons = lons_raw[lon_idx]

        self._lat_min = float(lats[0])
        self._lat_max = float(lats[-1])
        self._lon_min = float(lons[0])
        self._lon_max = float(lons[-1])
        self._p_min = float(levels[0])
        self._p_max = float(levels[-1])
        self._time_min = np.datetime64(int(times[0]), "ns")
        self._time_max = np.datetime64(int(times[-1]), "ns")

        axes = (times, levels, lats, lons)

        def _sorted_values(var_name: str) -> np.ndarray:
            """Вернуть numpy-массив переменной, переупорядоченный по всем осям."""
            arr = ds[var_name].values.astype(np.float32)
            # ERA5 порядок осей: (time, pressure_level, latitude, longitude)
            return arr[np.ix_(t_idx, l_idx, lat_idx, lon_idx)]

        u_data = np.nan_to_num(_sorted_values(_WIND_U_NAME), nan=0.0, posinf=0.0, neginf=0.0)
        v_data = np.nan_to_num(_sorted_values(_WIND_V_NAME), nan=0.0, posinf=0.0, neginf=0.0)
        w_data = np.nan_to_num(_sorted_values(_WIND_W_NAME), nan=0.0, posinf=0.0, neginf=0.0)
        t_data = np.nan_to_num(_sorted_values(_WIND_T_NAME), nan=288.15, posinf=288.15, neginf=288.15)

        kw = {"method": "linear", "bounds_error": False}
        self._interp_u = RegularGridInterpolator(axes, u_data, fill_value=0.0, **kw)
        self._interp_v = RegularGridInterpolator(axes, v_data, fill_value=0.0, **kw)
        self._interp_w = RegularGridInterpolator(axes, w_data, fill_value=0.0, **kw)
        self._interp_t = RegularGridInterpolator(axes, t_data, fill_value=288.15, **kw)

    @property
    def time_min(self) -> np.datetime64:
        return self._time_min

    @property
    def time_max(self) -> np.datetime64:
        return self._time_max

    # ──────────────────── Фабрика ────────────────────

    @classmethod
    def from_file(cls, path: Path, origin_lat: float, origin_lon: float) -> WindInterpolator:
        """Открыть датасет и создать интерполятор."""
        return cls(ds=xr.open_dataset(path), origin_lat=origin_lat, origin_lon=origin_lon)

    # ──────────────────── Преобразование координат ────────────────────

    def _xy_to_latlon(self, x_m: np.ndarray, y_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Локальные метры (x, y) → (lat, lon) относительно origin."""
        m_per_lat = meters_per_deg_lat(self.origin_lat)
        m_per_lon = meters_per_deg_lon(self.origin_lat)
        lat = np.clip(self.origin_lat + y_m / m_per_lat, self._lat_min, self._lat_max)
        lon = np.clip(self.origin_lon + x_m / m_per_lon, self._lon_min, self._lon_max)
        return lat.astype(np.float64), lon.astype(np.float64)

    def _z_to_pressure(self, z_m: np.ndarray) -> np.ndarray:
        """Высота (м) → давление (гПа) с кламп к диапазону датасета."""
        p = altitude_to_pressure_hpa(z_m).astype(np.float64)
        return np.clip(p, min(self._p_min, self._p_max), max(self._p_min, self._p_max))

    def _time_to_float(self, time: np.ndarray) -> np.ndarray:
        """datetime64 → float64 (наносекунды) с кламп к диапазону датасета."""
        t_ns = np.asarray(time, dtype="datetime64[ns]").astype(np.float64)
        t_min = float(self._time_min.astype("datetime64[ns]").astype(np.float64))
        t_max = float(self._time_max.astype("datetime64[ns]").astype(np.float64))
        return np.clip(t_ns, t_min, t_max)

    # ──────────────────── Публичный API ────────────────────

    def vector_at(self, x: float, y: float, z: float, time: np.datetime64) -> WindSample:
        """Вектор ветра (u, v, w) м/с, температура (K) и давление (гПа) в заданной точке."""
        lat, lon = self._xy_to_latlon(np.array([x]), np.array([y]))
        level = self._z_to_pressure(np.array([z]))
        t = self._time_to_float(np.array([time]))

        pt = np.column_stack([t, level, lat, lon])
        u = np.float32(self._interp_u(pt)[0])
        v = np.float32(self._interp_v(pt)[0])
        w_omega = np.float32(self._interp_w(pt)[0])
        temp = np.float32(self._interp_t(pt)[0])
        w = np.float32(omega_to_w_mps(np.array([w_omega]), level, np.array([temp]))[0])
        return WindSample(u=u, v=v, w=w, temperature=temp, pressure=np.float32(level[0]))

    def batch_vector_at(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, time: np.ndarray) -> np.ndarray:
        """Векторы ветра (u, v, w) м/с для батча точек. Shape: (n, 3)."""
        lat, lon = self._xy_to_latlon(x, y)
        level = self._z_to_pressure(z)
        t = self._time_to_float(time)

        pt = np.column_stack([t, level, lat, lon])
        u = self._interp_u(pt).astype(np.float32)
        v = self._interp_v(pt).astype(np.float32)
        w_omega = self._interp_w(pt).astype(np.float32)
        temp = self._interp_t(pt).astype(np.float32)
        w = omega_to_w_mps(w_omega, level, temp).astype(np.float32)
        return np.stack([u, v, w], axis=-1)

    def close(self) -> None:
        self.ds.close()
