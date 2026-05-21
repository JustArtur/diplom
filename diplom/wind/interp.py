"""Интерполятор ветрового поля ERA5 (u, v, w) в локальных метровых координатах.

Тяжёлые массивы интерполятора кэшируются в отдельный `.npy`-файл и
подключаются через `np.memmap`, чтобы несколько процессов могли разделять
одну файловую копию данных вместо дублирования памяти.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from diplom.geo import altitude_to_pressure_hpa, altitude_to_pressure_hpa_scalar, meters_per_deg_lat, meters_per_deg_lon
from diplom.data.era5_paths import wind_cache_meta_path, wind_cache_value_path
from diplom.world import WorldBounds, world_bounds_from_axes
from diplom.wind.trilinear import RegularGrid4DSampler


# ──────────────────── Конвертация вертикальной скорости ────────────────────

_R_DRY_AIR = 287.058  # Дж/(кг·К)
_GRAVITY = 9.80665  # м/с²


def omega_to_w_mps_scalar(omega_pa_s: float, pressure_hpa: float, temperature_k: float) -> float:
    """Scalar-версия `omega_to_w_mps` для hot path (одна точка после интерполяции)."""
    p_pa = float(pressure_hpa) * 100.0
    safe_p = p_pa if p_pa > 1.0 else 1.0
    return -float(omega_pa_s) * _R_DRY_AIR * float(temperature_k) / (safe_p * _GRAVITY)


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

    r_dry = _R_DRY_AIR
    g = _GRAVITY

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


# ──────────────────── Кэш интерполятора ────────────────────

_CACHE_VALUE_SUFFIX = ".wind-cache.npy"
_CACHE_META_SUFFIX = ".wind-cache.json"


def _legacy_cache_value_path(source_path: Path) -> Path:
    return source_path.with_name(f"{source_path.name}{_CACHE_VALUE_SUFFIX}")


def _legacy_cache_meta_path(source_path: Path) -> Path:
    return source_path.with_name(f"{source_path.name}{_CACHE_META_SUFFIX}")


def _cache_value_path(source_path: Path) -> Path:
    return wind_cache_value_path(source_path)


def _cache_meta_path(source_path: Path) -> Path:
    return wind_cache_meta_path(source_path)


def _source_signature(source_path: Path) -> dict[str, int | str]:
    stat = source_path.stat()
    return {
        "source_path": str(source_path.resolve()),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "source_size": int(stat.st_size),
    }


def _is_cache_valid(source_path: Path, value_path: Path, meta_path: Path) -> bool:
    if not value_path.exists() or not meta_path.exists():
        return False

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    try:
        signature = _source_signature(source_path)
    except OSError:
        return False

    required_keys = {"source_path", "source_mtime_ns", "source_size", "grid_shape"}
    if not required_keys.issubset(meta):
        return False

    return (
        str(meta["source_path"]) == signature["source_path"]
        and int(meta["source_mtime_ns"]) == signature["source_mtime_ns"]
        and int(meta["source_size"]) == signature["source_size"]
    )


def _write_cache(
    *,
    source_path: Path,
    value_path: Path,
    meta_path: Path,
    time_axis_ns: np.ndarray,
    pressure_axis_hpa: np.ndarray,
    latitude_axis_deg: np.ndarray,
    longitude_axis_deg: np.ndarray,
    u_data: np.ndarray,
    v_data: np.ndarray,
    w_data: np.ndarray,
    t_data: np.ndarray,
) -> None:
    """Собрать кэш интерполятора в одном memmap-файле."""
    token = uuid.uuid4().hex
    tmp_value_path = value_path.with_name(f".{value_path.name}.{token}.tmp")
    tmp_meta_path = meta_path.with_name(f".{meta_path.name}.{token}.tmp")

    try:
        values = np.lib.format.open_memmap(
            tmp_value_path,
            mode="w+",
            dtype=np.float32,
            shape=(4, *u_data.shape),
        )
        values[0] = u_data
        values[1] = v_data
        values[2] = w_data
        values[3] = t_data
        values.flush()
        del values

        meta = {
            **_source_signature(source_path),
            "grid_shape": [int(dim) for dim in u_data.shape],
            "time_axis_ns": [int(value) for value in time_axis_ns],
            "pressure_axis_hpa": [float(value) for value in pressure_axis_hpa],
            "latitude_axis_deg": [float(value) for value in latitude_axis_deg],
            "longitude_axis_deg": [float(value) for value in longitude_axis_deg],
        }
        tmp_meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        os.replace(tmp_value_path, value_path)
        os.replace(tmp_meta_path, meta_path)
    except Exception:
        for path in (tmp_value_path, tmp_meta_path):
            try:
                path.unlink()
            except OSError:
                pass
        raise


def ensure_wind_interpolator_cache(source_path: Path) -> tuple[Path, Path]:
    """Обеспечить наличие валидного кэша интерполятора в ``data/cache/wind/``.

    Не создаёт объект интерполяции — только записывает memmap-пакет и метаданные,
    если кэша ещё нет или источник изменился.

    Возвращает пути к файлам значений и метаданных для ``WindInterpolator``.
    """
    value_path = _cache_value_path(source_path)
    meta_path = _cache_meta_path(source_path)
    if _is_cache_valid(source_path, value_path, meta_path):
        return value_path, meta_path

    legacy_value = _legacy_cache_value_path(source_path)
    legacy_meta = _legacy_cache_meta_path(source_path)
    if _is_cache_valid(source_path, legacy_value, legacy_meta):
        value_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(legacy_value, value_path)
        shutil.move(legacy_meta, meta_path)
        return value_path, meta_path

    with xr.open_dataset(source_path) as ds:
        times_raw = ds[_TIME_NAME].values.astype("datetime64[ns]")
        levels_raw = ds[_LEVEL_NAME].values.astype(np.float64)
        lats_raw = ds[_LAT_NAME].values.astype(np.float64)
        lons_raw = ds[_LON_NAME].values.astype(np.float64)

        t_idx = np.argsort(times_raw)
        l_idx = np.argsort(levels_raw)
        lat_idx = np.argsort(lats_raw)
        lon_idx = np.argsort(lons_raw)

        time_axis_ns = times_raw[t_idx].astype("datetime64[ns]").astype(np.int64)
        pressure_axis_hpa = levels_raw[l_idx].astype(np.float64)
        latitude_axis_deg = lats_raw[lat_idx].astype(np.float64)
        longitude_axis_deg = lons_raw[lon_idx].astype(np.float64)

        def _sorted_values(var_name: str) -> np.ndarray:
            arr = ds[var_name].values.astype(np.float32)
            return arr[np.ix_(t_idx, l_idx, lat_idx, lon_idx)]

        u_data = np.nan_to_num(_sorted_values(_WIND_U_NAME), nan=0.0, posinf=0.0, neginf=0.0)
        v_data = np.nan_to_num(_sorted_values(_WIND_V_NAME), nan=0.0, posinf=0.0, neginf=0.0)
        w_data = np.nan_to_num(_sorted_values(_WIND_W_NAME), nan=0.0, posinf=0.0, neginf=0.0)
        t_data = np.nan_to_num(_sorted_values(_WIND_T_NAME), nan=288.15, posinf=288.15, neginf=288.15)

    value_path.parent.mkdir(parents=True, exist_ok=True)
    _write_cache(
        source_path=source_path,
        value_path=value_path,
        meta_path=meta_path,
        time_axis_ns=time_axis_ns,
        pressure_axis_hpa=pressure_axis_hpa,
        latitude_axis_deg=latitude_axis_deg,
        longitude_axis_deg=longitude_axis_deg,
        u_data=u_data,
        v_data=v_data,
        w_data=w_data,
        t_data=t_data,
    )
    return value_path, meta_path


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

    При инициализации данные загружаются из кэша на диске, а не из NetCDF,
    поэтому несколько subprocess-ов могут разделять одну файловую копию
    массивов через `memmap`.
    """

    data: np.ndarray
    env_idx: int | None
    origin_lat: float
    origin_lon: float
    time_axis_ns: np.ndarray
    pressure_axis_hpa: np.ndarray
    latitude_axis_deg: np.ndarray
    longitude_axis_deg: np.ndarray
    world_bounds: WorldBounds = field(init=False, repr=False)

    # scipy — для совместимости; hot path использует _grid_sampler
    _interp: RegularGridInterpolator = field(init=False, repr=False)
    _grid_sampler: RegularGrid4DSampler = field(init=False, repr=False)
    _time_axis_float: np.ndarray = field(init=False, repr=False)
    _m_per_lat: float = field(init=False, repr=False)
    _m_per_lon: float = field(init=False, repr=False)
    _time_min_float: float = field(init=False, repr=False)
    _time_max_float: float = field(init=False, repr=False)
    _p_clip_min: float = field(init=False, repr=False)
    _p_clip_max: float = field(init=False, repr=False)
    _pt_buf: np.ndarray = field(init=False, repr=False)
    _lat_min: float = field(init=False, repr=False)
    _lat_max: float = field(init=False, repr=False)
    _lon_min: float = field(init=False, repr=False)
    _lon_max: float = field(init=False, repr=False)
    _p_min: float = field(init=False, repr=False)
    _p_max: float = field(init=False, repr=False)
    _time_min: np.datetime64 = field(init=False, repr=False)
    _time_max: np.datetime64 = field(init=False, repr=False)
    _warned_dataset_bounds: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data)
        self.time_axis_ns = np.asarray(self.time_axis_ns, dtype=np.int64)
        self.pressure_axis_hpa = np.asarray(self.pressure_axis_hpa, dtype=np.float64)
        self.latitude_axis_deg = np.asarray(self.latitude_axis_deg, dtype=np.float64)
        self.longitude_axis_deg = np.asarray(self.longitude_axis_deg, dtype=np.float64)
        self._time_axis_float = self.time_axis_ns.astype(np.float64)
        self._warned_dataset_bounds = False
        self._build_scipy_interpolators()

    def _build_scipy_interpolators(self) -> None:
        """Собрать один vector-valued `RegularGridInterpolator` поверх общего кэша."""
        self._lat_min = float(self.latitude_axis_deg[0])
        self._lat_max = float(self.latitude_axis_deg[-1])
        self._lon_min = float(self.longitude_axis_deg[0])
        self._lon_max = float(self.longitude_axis_deg[-1])
        self.world_bounds = world_bounds_from_axes(
            self.latitude_axis_deg,
            self.longitude_axis_deg,
            origin_lat=self.origin_lat,
            origin_lon=self.origin_lon,
            pressure_axis_hpa=self.pressure_axis_hpa,
        )
        self._p_min = float(self.pressure_axis_hpa[0])
        self._p_max = float(self.pressure_axis_hpa[-1])
        self._time_min = np.datetime64(int(self.time_axis_ns[0]), "ns")
        self._time_max = np.datetime64(int(self.time_axis_ns[-1]), "ns")

        axes = (
            self._time_axis_float,
            self.pressure_axis_hpa,
            self.latitude_axis_deg,
            self.longitude_axis_deg,
        )

        # (4, T, P, Lat, Lon) → (T, P, Lat, Lon, 4): view без копии memmap.
        values = np.moveaxis(self.data, 0, -1)
        fill_value = np.array([0.0, 0.0, 0.0, 288.15], dtype=np.float64)
        self._interp = RegularGridInterpolator(
            axes,
            values,
            method="linear",
            bounds_error=False,
            fill_value=fill_value,
        )
        self._grid_sampler = RegularGrid4DSampler(
            values=values,
            time_axis=self._time_axis_float,
            pressure_axis=self.pressure_axis_hpa,
            lat_axis=self.latitude_axis_deg,
            lon_axis=self.longitude_axis_deg,
        )

        self._m_per_lat = meters_per_deg_lat(self.origin_lat)
        self._m_per_lon = meters_per_deg_lon(self.origin_lat)
        self._time_min_float = float(self._time_min.astype("datetime64[ns]").astype(np.float64))
        self._time_max_float = float(self._time_max.astype("datetime64[ns]").astype(np.float64))
        self._p_clip_min = min(self._p_min, self._p_max)
        self._p_clip_max = max(self._p_min, self._p_max)
        self._pt_buf = np.empty((1, 4), dtype=np.float64)

    @property
    def time_min(self) -> np.datetime64:
        return self._time_min

    @property
    def time_max(self) -> np.datetime64:
        return self._time_max

    # ──────────────────── Фабрика ────────────────────

    @classmethod
    def from_file(
        cls,
        path: Path,
        env_idx: int | None = None,
        origin_lat: float | None = None,
        origin_lon: float | None = None,
    ) -> WindInterpolator:
        """Открыть или создать кэш и затем построить интерполятор поверх него."""
        value_path, meta_path = ensure_wind_interpolator_cache(path)
        return cls._from_cache(
            value_path=value_path,
            meta_path=meta_path,
            env_idx=env_idx,
            origin_lat=origin_lat,
            origin_lon=origin_lon,
        )

    @classmethod
    def _from_cache(
        cls,
        *,
        value_path: Path,
        meta_path: Path,
        env_idx: int | None,
        origin_lat: float | None,
        origin_lon: float | None,
    ) -> WindInterpolator:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        data = np.load(value_path, mmap_mode="r")

        expected_shape = tuple(int(dim) for dim in meta["grid_shape"])
        if data.shape != (4, *expected_shape):
            raise ValueError(
                f"Invalid wind cache shape: got {data.shape}, expected {(4, *expected_shape)}"
            )

        origin_lat_value = (
            float(origin_lat)
            if origin_lat is not None
            else float(meta["latitude_axis_deg"][0])
        )
        origin_lon_value = (
            float(origin_lon)
            if origin_lon is not None
            else float(meta["longitude_axis_deg"][0])
        )

        return cls(
            data=data,
            env_idx=env_idx,
            origin_lat=origin_lat_value,
            origin_lon=origin_lon_value,
            time_axis_ns=np.asarray(meta["time_axis_ns"], dtype=np.int64),
            pressure_axis_hpa=np.asarray(meta["pressure_axis_hpa"], dtype=np.float64),
            latitude_axis_deg=np.asarray(meta["latitude_axis_deg"], dtype=np.float64),
            longitude_axis_deg=np.asarray(meta["longitude_axis_deg"], dtype=np.float64),
        )

    # ──────────────────── Преобразование координат ────────────────────

    def _xy_to_latlon(self, x_m: np.ndarray, y_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Локальные метры (x, y) → (lat, lon) относительно origin."""
        m_per_lat = meters_per_deg_lat(self.origin_lat)
        m_per_lon = meters_per_deg_lon(self.origin_lat)
        raw_lat = self.origin_lat + y_m / m_per_lat
        raw_lon = self.origin_lon + x_m / m_per_lon
        lat = np.clip(raw_lat, self._lat_min, self._lat_max)
        lon = np.clip(raw_lon, self._lon_min, self._lon_max)

        # if not self._warned_dataset_bounds and (
        #     np.any(raw_lat != lat) or np.any(raw_lon != lon)
        # ):
        #     self._warned_dataset_bounds = True
        #     env_label = f"env_{self.env_idx:03d}" if self.env_idx is not None else "env"
        #     warnings.warn(
        #         (
        #             f"[{env_label}] Позиция аэростата вышла за границы ERA5-датасета; "
        #             "координаты будут клампиться к доступному диапазону "
        #             f"lat=[{self._lat_min:.3f}, {self._lat_max:.3f}], "
        #             f"lon=[{self._lon_min:.3f}, {self._lon_max:.3f}]."
        #         ),
        #         RuntimeWarning,
        #         stacklevel=3,
        #     )

        return lat.astype(np.float64), lon.astype(np.float64)

    def _xy_to_latlon_scalar(self, x_m: float, y_m: float) -> tuple[float, float]:
        """Локальные метры (x, y) → (lat, lon) для одной точки без numpy-массивов."""
        raw_lat = self.origin_lat + y_m / self._m_per_lat
        raw_lon = self.origin_lon + x_m / self._m_per_lon
        lat = min(max(raw_lat, self._lat_min), self._lat_max)
        lon = min(max(raw_lon, self._lon_min), self._lon_max)

        # if not self._warned_dataset_bounds and (raw_lat != lat or raw_lon != lon):
        #     self._warned_dataset_bounds = True
        #     env_label = f"env_{self.env_idx:03d}" if self.env_idx is not None else "env"
        #     warnings.warn(
        #         (
        #             f"[{env_label}] Позиция аэростата вышла за границы ERA5-датасета; "
        #             "координаты будут клампиться к доступному диапазону "
        #             f"lat=[{self._lat_min:.3f}, {self._lat_max:.3f}], "
        #             f"lon=[{self._lon_min:.3f}, {self._lon_max:.3f}]."
        #         ),
        #         RuntimeWarning,
        #         stacklevel=3,
        #     )

        return lat, lon

    def _z_to_pressure_scalar(self, z_m: float) -> float:
        """Высота (м) → давление (гПа) для одной точки."""
        p = altitude_to_pressure_hpa_scalar(z_m)
        return min(max(p, self._p_clip_min), self._p_clip_max)

    def _time_to_float_scalar(self, time: np.datetime64) -> float:
        """datetime64 → float64 (наносекунды) для одной точки."""
        t_ns = float(np.datetime64(time, "ns").astype(np.float64))
        return min(max(t_ns, self._time_min_float), self._time_max_float)

    def _z_to_pressure(self, z_m: np.ndarray) -> np.ndarray:
        """Высота (м) → давление (гПа) с кламп к диапазону датасета."""
        p = altitude_to_pressure_hpa(z_m).astype(np.float64)
        return np.clip(p, self._p_clip_min, self._p_clip_max)

    def _time_to_float(self, time: np.ndarray) -> np.ndarray:
        """datetime64 → float64 (наносекунды) с кламп к диапазону датасета."""
        t_ns = np.asarray(time, dtype="datetime64[ns]").astype(np.float64)
        return np.clip(t_ns, self._time_min_float, self._time_max_float)

    # ──────────────────── Публичный API ────────────────────

    def vector_at(self, x: float, y: float, z: float, time: np.datetime64) -> WindSample:
        """Вектор ветра (u, v, w) м/с, температура (K) и давление (гПа) в заданной точке."""
        lat, lon = self._xy_to_latlon_scalar(x, y)
        level = self._z_to_pressure_scalar(z)
        t = self._time_to_float_scalar(time)

        u, v, w_omega, temp = self._grid_sampler.sample(t, level, lat, lon)
        w = omega_to_w_mps_scalar(float(w_omega), level, float(temp))
        return WindSample(u=float(u), v=float(v), w=w, temperature=float(temp), pressure=level)

    def batch_vector_at(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, time: np.ndarray) -> np.ndarray:
        """Векторы ветра (u, v, w) м/с для батча точек. Shape: (n, 3)."""
        lat, lon = self._xy_to_latlon(x, y)
        level = self._z_to_pressure(z)
        t = self._time_to_float(time)

        pt = np.column_stack([t, level, lat, lon])
        u, v, w_omega, _temp = np.moveaxis(self._interp(pt).astype(np.float32), -1, 0)
        w = omega_to_w_mps(w_omega, level, _temp).astype(np.float32)
        return np.stack([u, v, w], axis=-1)

    def close(self) -> None:
        mmap = getattr(self.data, "_mmap", None)
        if mmap is not None:
            mmap.close()
