"""3D-визуализация поля ветра ERA5 (Plotly HTML).

Строит интерактивный 3D-граф с конусами (Cone), отображающими направление и
скорость ветра на всех высотах, широтах и долготах для выбранного временного
среза ERA5-датасета.

Публичный API:
  list_available_times(path)               → list[np.datetime64]
  load_wind_slice(path, time_target)       → WindSlice
  build_wind_figure(slice_, **opts)        → go.Figure
  save_figure(fig, path)                   → standalone HTML

Пакетная отрисовка всех датасетов из ``data/`` — команда CLI ``diplom wind-viz``.
"""

from __future__ import annotations

import colorsys
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go
import xarray as xr
from scipy.interpolate import interp1d

from diplom.geo import altitude_to_pressure_hpa, meters_per_deg_lat, meters_per_deg_lon
from diplom.world import world_bounds_from_axes
from diplom.wind.interp import omega_to_w_mps

# ──────────────────── Имена переменных ERA5 ────────────────────

_U_VAR = "u"
_V_VAR = "v"
_W_VAR = "w"
_T_VAR = "t"
_LON_DIM = "longitude"
_LAT_DIM = "latitude"
_LEVEL_DIM = "pressure_level"
_TIME_DIM = "valid_time"

# Циклическая HSL-палитра по азимуту: 0° и 360° совпадают, противоположные направления контрастны.
DEFAULT_CONE_AZIMUTH_COLORSCALE: list[list[float | str]] = [
    [0.0, "hsl(0, 82%, 58%)"],
    [0.125, "hsl(45, 82%, 58%)"],
    [0.25, "hsl(90, 82%, 58%)"],
    [0.375, "hsl(135, 82%, 58%)"],
    [0.5, "hsl(180, 82%, 58%)"],
    [0.625, "hsl(225, 82%, 58%)"],
    [0.75, "hsl(270, 82%, 58%)"],
    [0.875, "hsl(315, 82%, 58%)"],
    [1.0, "hsl(0, 82%, 58%)"],
]

# Число секторов по азимуту: Plotly Cone красит только по ‖V‖, поэтому каждый сектор — flat colorscale.
CONE_AZIMUTH_BINS = 36


def _horizontal_azimuth_deg_east_north(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Горизонтальный азимут ветра: 0° = E, 90° = N, диапазон [0, 360)."""
    return np.mod(np.degrees(np.arctan2(np.asarray(v, dtype=np.float64), np.asarray(u, dtype=np.float64))), 360.0)


def _parse_hsl(color: str) -> tuple[float, float, float]:
    match = re.match(r"hsl\(\s*([0-9.]+)\s*,\s*([0-9.]+)%\s*,\s*([0-9.]+)%\s*\)", color.strip())
    if match is None:
        raise ValueError(f"Ожидался hsl(...), получено {color!r}.")
    h, s_pct, l_pct = match.groups()
    return float(h), float(s_pct) / 100.0, float(l_pct) / 100.0


def _hsl_to_rgb_str(h: float, s: float, lightness: float) -> str:
    r, g, b = colorsys.hls_to_rgb((h % 360.0) / 360.0, lightness, s)
    return f"rgb({int(round(r * 255))},{int(round(g * 255))},{int(round(b * 255))})"


def _interp_hsl(a: tuple[float, float, float], b: tuple[float, float, float], frac: float) -> str:
    h0, s0, l0 = a
    h1, s1, l1 = b
    dh = h1 - h0
    if dh > 180.0:
        dh -= 360.0
    elif dh < -180.0:
        dh += 360.0
    h = h0 + dh * frac
    s = s0 + (s1 - s0) * frac
    lightness = l0 + (l1 - l0) * frac
    return _hsl_to_rgb_str(h, s, lightness)


def _sample_azimuth_color(
    colorscale: str | list[list[float | str]],
    azimuth_deg: float,
    *,
    reversescale: bool,
) -> str:
    t = float(np.mod(azimuth_deg, 360.0)) / 360.0
    if reversescale:
        t = 1.0 - t
    if isinstance(colorscale, str):
        return pc.sample_colorscale(colorscale, [t])[0]
    stops = sorted(colorscale, key=lambda item: float(item[0]))
    if not stops:
        raise ValueError("colorscale не должен быть пустым.")
    if t <= float(stops[0][0]):
        color = str(stops[0][1])
    elif t >= float(stops[-1][0]):
        color = str(stops[-1][1])
    else:
        color = str(stops[-1][1])
        for left, right in zip(stops, stops[1:]):
            p0, c0 = float(left[0]), str(left[1])
            p1, c1 = float(right[0]), str(right[1])
            if p0 <= t <= p1:
                frac = (t - p0) / (p1 - p0) if p1 > p0 else 0.0
                if c0.startswith("hsl(") and c1.startswith("hsl("):
                    color = _interp_hsl(_parse_hsl(c0), _parse_hsl(c1), frac)
                else:
                    color = pc.find_intermediate_color(c0, c1, frac)
                break
    return color


def _flat_colorscale_for_azimuth(
    azimuth_deg: float,
    colorscale: str | list[list[float | str]],
    *,
    reversescale: bool,
) -> list[list[float | str]]:
    """Flat colorscale для одного конуса: цвет по азимуту, длина вектора — только скорость."""
    rgb = _sample_azimuth_color(colorscale, azimuth_deg, reversescale=reversescale)
    return [[0.0, rgb], [1.0, rgb]]


def _compressed_speed_mag(
    speed: np.ndarray,
    *,
    floor: float,
    power: float,
) -> np.ndarray:
    """Сжать диапазон нормы вектора конуса: слабая зависимость от скорости (узкий разброс размеров)."""
    if floor <= 0 or floor >= 1:
        raise ValueError("cone_speed_floor должен быть в (0, 1).")
    if power <= 0:
        raise ValueError("cone_speed_power должен быть > 0.")
    s = np.asarray(speed, dtype=np.float64)
    if s.size == 0:
        return s
    mx = float(np.max(s))
    if mx < 1e-12:
        return np.ones_like(s)
    t = np.clip(s / mx, 0.0, 1.0)
    return floor + (1.0 - floor) * np.power(t, power)




def _ensure_scene_axis_range(lo: float, hi: float, *, min_half_width_m: float = 500.0) -> tuple[float, float]:
    """Plotly не любит совпадающие min/max; для решётки из одной точки расширяем полуширину."""
    center = (lo + hi) * 0.5
    half = max((hi - lo) * 0.5, min_half_width_m)
    return center - half, center + half


def _altitude_targets_m(h_m: np.ndarray, step_m: float) -> np.ndarray:
    """Равномерная сетка высоты (м) от минимума к максимуму в пределах нативных уровней среза."""
    if step_m <= 0:
        raise ValueError("stride_altitude_m (шаг по высоте, м) должен быть > 0.")
    h = np.sort(np.unique(np.asarray(h_m, dtype=np.float64)))
    if h.size == 0:
        raise ValueError("Датасет не содержит уровней по высоте.")
    lo, hi = float(h[0]), float(h[-1])
    if hi - lo < 1e-6:
        return np.array([lo], dtype=np.float64)
    pts = np.arange(lo, hi, step_m, dtype=np.float64)
    if pts.size == 0 or abs(float(pts[-1]) - hi) > max(1e-6, 1e-3 * step_m):
        pts = np.append(pts, hi)
    else:
        pts[-1] = hi
    return pts


def _interp_wind_vertical_to_altitude(
    slice_: WindSlice,
    h_targets_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Линейная интерполяция u, v, w между уровнями ERA5 на заданные высоты (м)."""
    order = np.argsort(slice_.altitude.astype(np.float64))
    h_asc = slice_.altitude[order].astype(np.float64)
    u_asc = slice_.u[order, ...].astype(np.float64)
    v_asc = slice_.v[order, ...].astype(np.float64)
    w_asc = slice_.w[order, ...].astype(np.float64)
    if h_asc.size == 1:
        shp = (len(h_targets_m),) + tuple(u_asc.shape[1:])
        return (
            np.broadcast_to(u_asc[0], shp).astype(np.float32).copy(),
            np.broadcast_to(v_asc[0], shp).astype(np.float32).copy(),
            np.broadcast_to(w_asc[0], shp).astype(np.float32).copy(),
        )
    ht = np.clip(h_targets_m.astype(np.float64), h_asc[0], h_asc[-1])
    kw = dict(axis=0, kind="linear", bounds_error=False, fill_value=(u_asc[0], u_asc[-1]))
    u_i = interp1d(h_asc, u_asc, **kw)(ht).astype(np.float32)
    v_i = interp1d(h_asc, v_asc, **kw)(ht).astype(np.float32)
    w_i = interp1d(h_asc, w_asc, **kw)(ht).astype(np.float32)
    return u_i, v_i, w_i


# ──────────────────── Конвертация давление → высота ────────────────────


def _pressure_to_altitude_m(pressure_hpa: np.ndarray) -> np.ndarray:
    """Обратная барометрическая формула ISA: давление (гПа) → высота (м).

    h = (T₀/L) · (1 − (p/p₀) ^ (1/exponent))
    где exponent = g·M / (R·L) ≈ 5.2559.
    """
    p0 = 1013.25
    T0 = 288.15
    L = 0.0065
    g = 9.80665
    R = 8.31447
    M = 0.0289644
    exp = (g * M) / (R * L)
    ratio = np.clip(np.asarray(pressure_hpa, dtype=np.float64) / p0, 1e-9, 1.0)
    return (T0 / L) * (1.0 - np.power(ratio, 1.0 / exp))


def _default_origin(slice_: "WindSlice") -> tuple[float, float]:
    """Брать опорную точку из фактической юго-западной границы среза."""
    return float(slice_.origin_lat), float(slice_.origin_lon)


def _lonlat_to_local_meters(
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    origin_lon: float,
    origin_lat: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Перевести долготу/широту в локальные метры через meters-per-degree."""
    m_per_lat = meters_per_deg_lat(origin_lat)
    m_per_lon = meters_per_deg_lon(origin_lat)
    x_m = (np.asarray(lon, dtype=np.float64) - origin_lon) * m_per_lon
    y_m = (np.asarray(lat, dtype=np.float64) - origin_lat) * m_per_lat
    return x_m.astype(np.float32), y_m.astype(np.float32)


# ──────────────────── Срез ветра ────────────────────


@dataclass
class WindSlice:
    """Ветровое поле на одном временном шаге ERA5."""

    lon: np.ndarray       # (nlon,)  градусы
    lat: np.ndarray       # (nlat,)  градусы
    origin_lon: float     # опорная долгота для локальной системы координат
    origin_lat: float     # опорная широта для локальной системы координат
    pressure: np.ndarray  # (nlevel,)  гПа (убывает → высота растёт)
    altitude: np.ndarray  # (nlevel,)  метры (возрастает)
    time: np.datetime64   # временна́я метка среза

    # Ветровые компоненты — shape (nlevel, nlat, nlon)
    u: np.ndarray     # западно-восточная, м/с
    v: np.ndarray     # южно-северная, м/с
    w: np.ndarray     # вертикальная (omega→м/с), м/с (+ = вверх)
    speed_h: np.ndarray   # горизонтальная скорость sqrt(u²+v²), м/с
    speed_3d: np.ndarray  # полная скорость sqrt(u²+v²+w²), м/с


# ──────────────────── Список доступных временных меток ────────────────────


def list_available_times(path: Path) -> List[np.datetime64]:
    """Вернуть список временных меток, доступных в ERA5-файле."""
    with xr.open_dataset(path) as ds:
        times = ds[_TIME_DIM].values.astype("datetime64[s]")
    return [np.datetime64(t, "s") for t in times]


# ──────────────────── Загрузка среза ────────────────────


def load_wind_slice(path: Path, time_target: np.datetime64) -> WindSlice:
    """Загрузить ветровой срез ERA5 на ближайшем к `time_target` шаге.

    Args:
        path: путь к NetCDF-файлу ERA5.
        time_target: желаемая временна́я метка; будет выбран ближайший шаг.

    Returns:
        WindSlice с конвертированными ветровыми компонентами.
    """
    ds = xr.open_dataset(path)
    try:
        times_ns = ds[_TIME_DIM].values.astype("datetime64[ns]")
        target_ns = np.datetime64(time_target, "ns")
        time_idx = int(np.argmin(np.abs(times_ns - target_ns)))
        actual_time = times_ns[time_idx].astype("datetime64[s]")

        ds_t = ds.isel({_TIME_DIM: time_idx})

        lon = ds_t[_LON_DIM].values.astype(np.float32)
        lat = ds_t[_LAT_DIM].values.astype(np.float32)
        pressure = ds_t[_LEVEL_DIM].values.astype(np.float32)

        # Сортируем уровни: давление убывает (высота растёт)
        p_order = np.argsort(pressure)[::-1]
        pressure = pressure[p_order]

        u_raw = ds_t[_U_VAR].values.astype(np.float32)[p_order]  # (nlevel, nlat, nlon)
        v_raw = ds_t[_V_VAR].values.astype(np.float32)[p_order]
        w_omega = ds_t[_W_VAR].values.astype(np.float32)[p_order]
        t_raw = ds_t[_T_VAR].values.astype(np.float32)[p_order]

        # omega (Па/с) → вертикальная скорость w (м/с)
        p_3d = pressure[:, np.newaxis, np.newaxis] * np.ones_like(u_raw)
        w_raw = omega_to_w_mps(w_omega, p_3d, t_raw)

        u = np.nan_to_num(u_raw, nan=0.0, posinf=0.0, neginf=0.0)
        v = np.nan_to_num(v_raw, nan=0.0, posinf=0.0, neginf=0.0)
        w = np.nan_to_num(w_raw, nan=0.0, posinf=0.0, neginf=0.0)

        altitude = _pressure_to_altitude_m(pressure).astype(np.float32)
        speed_h = np.sqrt(u**2 + v**2).astype(np.float32)
        speed_3d = np.sqrt(u**2 + v**2 + w**2).astype(np.float32)

        return WindSlice(
            lon=lon,
            lat=lat,
            origin_lon=float(lon.min()),
            origin_lat=float(lat.min()),
            pressure=pressure,
            altitude=altitude,
            time=np.datetime64(actual_time, "s"),
            u=u,
            v=v,
            w=w,
            speed_h=speed_h,
            speed_3d=speed_3d,
        )
    finally:
        ds.close()


# ──────────────────── Построение 3D-графика ────────────────────


def build_wind_figure(
    slice_: WindSlice,
    title: Optional[str] = None,
    stride_lon: int = 1,
    stride_lat: int = 1,
    stride_altitude_m: float = 500.0,
    w_scale: float = 0.0,
    cone_sizeref: float = 20,
    cone_azimuth_colorscale: Optional[str | list[list[float | str]]] = None,
    cone_azimuth_reversescale: bool = False,
    cone_speed_floor: float = 0.38,
    cone_speed_power: float = 0.42,
    calm_speed_mps: float = 0.12,
    altitude_unit: str = "km",
    origin_lat: Optional[float] = None,
    origin_lon: Optional[float] = None,
) -> go.Figure:
    """Построить интерактивный 3D-граф поля ветра ERA5.

    Конусы (Cone) кодируют:
      • Направление острия  → направление ветра (u, v, w·w_scale).
      • Цвет               → горизонтальный азимут **arctan2(v, u)** в градусах (0° = E, 90° = N) с
                             **циклической** палитрой: близкие направления (напр. 350° и 30°) дают близкий цвет,
                             противоположные (≈180°) — максимально различный.
      • Размер (слабо)     → горизонтальная скорость через сжатый множитель `cone_speed_*`; штиль — отдельно, серым.
      • Tooltip            → u, v, w (реальные), скорость, давление, высота, азимут.

    Args:
        slice_:       срез ERA5, полученный через `load_wind_slice`.
        title:        заголовок графика.
        stride_lon:   прореживание по долготе (1 = брать каждую точку).
        stride_lat:   прореживание по широте.
        stride_altitude_m: шаг сетки отрисовки по высоте в **метрах** (равномерно по оси Z).
                           Поля u, v, w линейно интерполируются между исходными уровнями ERA5 по высоте.
        w_scale:      масштаб вертикальной компоненты для наглядности.
        cone_sizeref: общий масштаб размера конусов Plotly; меньше значение = больше конусы.
        cone_azimuth_colorscale: циклическая палитра [[0,color],…,[1,color]] или имя Plotly; по умолчанию HSL-колесо.
                                   Для азимута нужна палитра, где первый и последний цвет совпадают.
        cone_azimuth_reversescale: перевернуть именованную палитру (игнорируется для пользовательского списка).
        cone_speed_floor: нижняя граница множителя от нормализованной скорости (узкий разброс длин между точками).
        cone_speed_power: степень нормализованной горизонтальной скорости для этого множителя.
        calm_speed_mps: |V_h| ниже порога → серый отдельный trace (штиль).
        altitude_unit: «km» или «m» — единицы на оси Z.
        origin_lat/origin_lon: опорная точка для локальных метровых координат (границы сцены как у `WindInterpolator.world_bounds`).
    """
    # ── Вертикаль: равномерная сетка по высоте (м), интерполяция ветра по h ──
    h_targets = _altitude_targets_m(slice_.altitude, float(stride_altitude_m))
    u_full, v_full, w_full = _interp_wind_vertical_to_altitude(slice_, h_targets)
    alt_sub = h_targets.astype(np.float32)
    pressure_sub_hpa = altitude_to_pressure_hpa(alt_sub.astype(np.float64))
    speed_sub = np.sqrt(u_full**2 + v_full**2).astype(np.float32)

    # ── Прореживание по горизонтали ──────────────────────────────────────
    la_idx = np.arange(0, len(slice_.lat), max(1, stride_lat))
    lo_idx = np.arange(0, len(slice_.lon), max(1, stride_lon))

    u_sub = u_full[:, la_idx, :][:, :, lo_idx]
    v_sub = v_full[:, la_idx, :][:, :, lo_idx]
    w_sub = w_full[:, la_idx, :][:, :, lo_idx]
    speed_sub = speed_sub[:, la_idx, :][:, :, lo_idx]
    pressure_sub = pressure_sub_hpa[:, np.newaxis, np.newaxis] * np.ones_like(u_sub, dtype=np.float32)
    lat_sub = slice_.lat[la_idx]
    lon_sub = slice_.lon[lo_idx]

    default_origin_lat, default_origin_lon = _default_origin(slice_)
    origin_lat = float(origin_lat) if origin_lat is not None else default_origin_lat
    origin_lon = float(origin_lon) if origin_lon is not None else default_origin_lon

    world_bounds_xy = world_bounds_from_axes(
        np.asarray(slice_.lat, dtype=np.float64),
        np.asarray(slice_.lon, dtype=np.float64),
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        pressure_axis_hpa=np.asarray(slice_.pressure, dtype=np.float64),
    )
    x_lo, x_hi = _ensure_scene_axis_range(world_bounds_xy.x_min, world_bounds_xy.x_max)
    y_lo, y_hi = _ensure_scene_axis_range(world_bounds_xy.y_min, world_bounds_xy.y_max)
    z_lo_m, z_hi_m = _ensure_scene_axis_range(world_bounds_xy.z_min, world_bounds_xy.z_max, min_half_width_m=200.0)
    x_scene_range = [x_lo, x_hi]
    y_scene_range = [y_lo, y_hi]
    z_scene_range = (
        [z_lo_m / 1000.0, z_hi_m / 1000.0] if altitude_unit == "km" else [z_lo_m, z_hi_m]
    )

    # ── Координатная сетка (nlevel, nlat, nlon) ────────────────────────
    alt_3d = alt_sub[:, np.newaxis, np.newaxis] * np.ones_like(u_sub, dtype=np.float32)
    lat_3d = lat_sub[np.newaxis, :, np.newaxis] * np.ones_like(u_sub, dtype=np.float32)
    lon_3d = lon_sub[np.newaxis, np.newaxis, :] * np.ones_like(u_sub, dtype=np.float32)
    p_3d = pressure_sub
    x_3d, y_3d = _lonlat_to_local_meters(
        lon_3d,
        lat_3d,
        origin_lon=origin_lon,
        origin_lat=origin_lat,
    )

    z_values = alt_3d / 1000.0 if altitude_unit == "km" else alt_3d

    # ── Вектора для Cone: цвет = азимут; длина ≈ скорость ─────────
    x_arr = np.asarray(x_3d.ravel(), dtype=np.float64)
    y_arr = np.asarray(y_3d.ravel(), dtype=np.float64)
    z_arr = np.asarray(z_values.ravel(), dtype=np.float64)
    u_arr = u_sub.ravel().astype(np.float64)
    v_arr = v_sub.ravel().astype(np.float64)
    w_vis_arr = (w_sub * float(w_scale)).ravel().astype(np.float64)
    w_real = w_sub.ravel().astype(np.float64)
    speed_flat = speed_sub.ravel().astype(np.float64)
    p_flat = p_3d.ravel().astype(np.float64)
    alt_flat = alt_3d.ravel().astype(np.float64)
    n_pts = x_arr.shape[0]

    dn = np.sqrt(u_arr * u_arr + v_arr * v_arr + w_vis_arr * w_vis_arr)
    dn_safe = np.maximum(dn, 1e-12)
    ux = u_arr / dn_safe
    vy = v_arr / dn_safe
    wz = w_vis_arr / dn_safe

    mag = _compressed_speed_mag(
        speed_flat,
        floor=float(cone_speed_floor),
        power=float(cone_speed_power),
    )
    ref_speed = float(np.max(speed_flat)) if n_pts > 0 else 1.0
    ref_speed = max(ref_speed, float(calm_speed_mps) * 2.0, 1.0)

    calm = speed_flat < float(calm_speed_mps)
    active = ~calm
    azimuth_deg = _horizontal_azimuth_deg_east_north(u_arr, v_arr)

    scale_amp = ref_speed * mag
    east = ux * scale_amp
    north = vy * scale_amp
    up = wz * scale_amp

    azimuth_tickvals = [0.0, 90.0, 180.0, 270.0]
    azimuth_ticktext = ["E (0°)", "N (90°)", "W (180°)", "S (270°)"]
    cbar_title = "Направление ветра (азимут)<br><sub>0° = E, 90° = N · горизонтальное</sub>"

    cs_raw = cone_azimuth_colorscale if cone_azimuth_colorscale is not None else DEFAULT_CONE_AZIMUTH_COLORSCALE
    use_reverse = cone_azimuth_reversescale and isinstance(cs_raw, str)

    def hover_for_indices(idx: np.ndarray) -> list[str]:
        out: list[str] = []
        for i in idx:
            ii = int(i)
            spd = float(speed_flat[ii])
            compass_deg = float(azimuth_deg[ii])
            out.append(
                f"<b>x={x_arr[ii]:.0f} м, y={y_arr[ii]:.0f} м</b><br>"
                f"Опорная точка: lon={origin_lon:.2f}°, lat={origin_lat:.2f}°<br>"
                f"Давление: {p_flat[ii]:.2f} гПа<br>"
                f"Высота: {alt_flat[ii]/1000:.1f} км ({alt_flat[ii]:.0f} м)<br>"
                f"<b>u = {u_arr[ii]:.2f} м/с</b> (W→E)<br>"
                f"<b>v = {v_arr[ii]:.2f} м/с</b> (S→N)<br>"
                f"<b>w = {w_real[ii]:.4f} м/с</b> (↑+)<br>"
                f"<b>|V_h| = {spd:.2f} м/с</b><br>"
                f"<b>Гориз. азимут ≈ {compass_deg:.0f}°</b> (0°=E, 90°=N)<br>"
            )
        return out

    calm_rgb = "rgb(138,143,156)"

    z_label = "Высота, км" if altitude_unit == "km" else "Высота, м"
    x_label = "X, м"
    y_label = "Y, м"
    auto_title = (
        title
        if title is not None
        else f"Поле ветра ERA5 · {slice_.time}"
    )

    fig = go.Figure()

    idx_calm = np.flatnonzero(calm)
    if idx_calm.size > 0:
        flat_cs = [[0.0, calm_rgb], [1.0, calm_rgb]]
        fig.add_trace(
            go.Cone(
                x=x_arr[idx_calm].tolist(),
                y=y_arr[idx_calm].tolist(),
                z=z_arr[idx_calm].tolist(),
                u=east[idx_calm].tolist(),
                v=north[idx_calm].tolist(),
                w=up[idx_calm].tolist(),
                colorscale=flat_cs,
                cmin=0.0,
                cmax=1.0,
                sizemode="absolute",
                sizeref=float(cone_sizeref),
                anchor="tail",
                hovertemplate="%{text}<extra></extra>",
                text=hover_for_indices(idx_calm),
                name="Штиль",
                showlegend=False,
                showscale=False,
            )
        )

    idx_act = np.flatnonzero(active)
    if idx_act.size > 0:
        bin_width = 360.0 / float(CONE_AZIMUTH_BINS)
        azimuth_bins = (np.floor(azimuth_deg / bin_width).astype(np.intp)) % CONE_AZIMUTH_BINS
        for bin_idx in range(CONE_AZIMUTH_BINS):
            sel = idx_act[azimuth_bins[idx_act] == bin_idx]
            if sel.size == 0:
                continue
            bin_azimuth = (bin_idx + 0.5) * bin_width
            flat_cs = _flat_colorscale_for_azimuth(
                bin_azimuth,
                cs_raw,
                reversescale=use_reverse,
            )
            fig.add_trace(
                go.Cone(
                    x=x_arr[sel].tolist(),
                    y=y_arr[sel].tolist(),
                    z=z_arr[sel].tolist(),
                    u=east[sel].tolist(),
                    v=north[sel].tolist(),
                    w=up[sel].tolist(),
                    colorscale=flat_cs,
                    cmin=0.0,
                    cmax=1.0,
                    sizemode="absolute",
                    sizeref=float(cone_sizeref),
                    anchor="tail",
                    hovertemplate="%{text}<extra></extra>",
                    text=hover_for_indices(sel),
                    showlegend=False,
                    showscale=False,
                )
            )
        xd = float(x_scene_range[1]) - float(x_scene_range[0])
        yd = float(y_scene_range[1]) - float(y_scene_range[0])
        zd = float(z_scene_range[1]) - float(z_scene_range[0])
        eps = max(xd, yd, zd, 1.0) * 1e-6
        x0 = float(x_scene_range[0])
        y0 = float(y_scene_range[0])
        z0 = float(z_scene_range[0])
        fig.add_trace(
            go.Scatter3d(
                x=[x0, x0 + eps, x0 + 2 * eps, x0 + 3 * eps],
                y=[y0, y0, y0, y0],
                z=[z0, z0, z0, z0],
                mode="markers",
                marker=dict(
                    color=azimuth_tickvals,
                    cmin=0.0,
                    cmax=360.0,
                    colorscale=cs_raw,
                    reversescale=use_reverse,
                    size=np.full(4, 2.2),
                    opacity=0,
                    showscale=True,
                    colorbar=dict(
                        title=dict(
                            text=cbar_title,
                            side="right",
                            font=dict(color="#e9ecf5", size=12),
                        ),
                        tickvals=azimuth_tickvals,
                        ticktext=azimuth_ticktext,
                        len=0.65,
                        thickness=16,
                        outlinewidth=0,
                        bgcolor="rgba(22,26,42,0.92)",
                        tickfont=dict(color="#e9ecf5"),
                        x=1.02,
                    ),
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # ── Аннотации ────────────────────────────────────────────────────────
    annotations = []
    if w_scale != 0.0:
        annotations.append(
            dict(
                text=(
                    f"⚠ Вертикальная компонента w масштабирована ×{w_scale:.0f} "
                    "для наглядности стрелок"
                ),
                xref="paper", yref="paper",
                x=0.01, y=0.01,
                showarrow=False,
                font=dict(size=11, color="rgba(200,200,200,0.8)"),
                align="left",
            )
        )

    fig.update_layout(
        title=dict(text=auto_title, x=0.5, font=dict(size=15)),
        scene=dict(
            xaxis=dict(title=x_label, range=x_scene_range, autorange=False),
            yaxis=dict(title=y_label, range=y_scene_range, autorange=False),
            zaxis=dict(title=z_label, range=z_scene_range, autorange=False),
            bgcolor="rgba(10,10,30,1)",
            camera=dict(
                eye=dict(x=1.6, y=-1.6, z=1.0),
            ),
        ),
        annotations=annotations,
        margin=dict(l=0, r=90, b=30, t=60),
        template="plotly_dark",
        paper_bgcolor="rgba(15,15,25,1)",
    )

    return fig


# ──────────────────── Сохранение ────────────────────


def save_figure(fig: go.Figure, path: Path) -> None:
    """Сохранить фигуру как standalone HTML (Plotly CDN, без встроенного JS)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")


# ──────────────────── Пакетная отрисовка (процессы) ────────────────────


@dataclass(frozen=True, slots=True)
class WindPlotRenderJob:
    """Задание на построение одного HTML-графика (picklable для ProcessPool)."""

    dataset_path: Path
    output_dir: Path
    time_ns: int | None  # np.datetime64[ns]; None — первый шаг датасета
    stride_lon: int
    stride_lat: int
    stride_altitude_m: float
    w_scale: float


@dataclass(frozen=True, slots=True)
class WindPlotRenderResult:
    dataset_name: str
    plot_path: Path | None
    saved: bool
    log_lines: tuple[str, ...]
    error: str | None = None


def render_wind_plot_job(job: WindPlotRenderJob) -> WindPlotRenderResult:
    """Построить и сохранить график для одного датасета (отдельный процесс)."""
    from diplom.data.era5_paths import era5_dataset_title, wind_plot_html_path
    from diplom.world import world_bounds_from_axes

    name = job.dataset_path.name
    plot_path = wind_plot_html_path(job.dataset_path, job.output_dir)
    logs: list[str] = []

    if plot_path.exists():
        return WindPlotRenderResult(
            dataset_name=name,
            plot_path=plot_path,
            saved=False,
            log_lines=(f"Пропуск {name}: график уже есть → {plot_path}",),
        )

    try:
        if job.time_ns is None:
            available = list_available_times(job.dataset_path)
            if not available:
                return WindPlotRenderResult(
                    dataset_name=name,
                    plot_path=None,
                    saved=False,
                    log_lines=(),
                    error=f"{name}: нет временны́х шагов.",
                )
            target_time = available[0]
            logs.append(f"{name}: --time не задано, первый шаг {target_time}")
        else:
            target_time = np.datetime64(job.time_ns, "ns")

        logs.append(f"Загружаю срез ERA5: {job.dataset_path} @ {target_time} …")
        wind_slice = load_wind_slice(job.dataset_path, target_time)
        logs.append(
            f"Срез загружен · {name} · время={wind_slice.time} "
            f"· уровней={len(wind_slice.pressure)} "
            f"· lat={len(wind_slice.lat)} · lon={len(wind_slice.lon)}"
        )

        wb = world_bounds_from_axes(
            np.asarray(wind_slice.lat, dtype=np.float64),
            np.asarray(wind_slice.lon, dtype=np.float64),
            origin_lat=wind_slice.origin_lat,
            origin_lon=wind_slice.origin_lon,
            pressure_axis_hpa=np.asarray(wind_slice.pressure, dtype=np.float64),
        )
        logs.append(
            f"[wind-viz] {name} · X [{wb.x_min:.1f} … {wb.x_max:.1f}] м · "
            f"Y [{wb.y_min:.1f} … {wb.y_max:.1f}] м · "
            f"Z [{wb.z_min:.1f} … {wb.z_max:.1f}] м"
        )

        fig = build_wind_figure(
            wind_slice,
            title=era5_dataset_title(job.dataset_path),
            stride_lon=job.stride_lon,
            stride_lat=job.stride_lat,
            stride_altitude_m=job.stride_altitude_m,
            w_scale=job.w_scale,
        )
        save_figure(fig, plot_path)
        logs.append(f"График сохранён: {plot_path}")
        return WindPlotRenderResult(
            dataset_name=name,
            plot_path=plot_path,
            saved=True,
            log_lines=tuple(logs),
        )
    except Exception as exc:
        return WindPlotRenderResult(
            dataset_name=name,
            plot_path=None,
            saved=False,
            log_lines=tuple(logs),
            error=f"{name}: {exc}",
        )


def render_wind_plots(
    jobs: Sequence[WindPlotRenderJob],
    *,
    workers: int = 1,
) -> list[WindPlotRenderResult]:
    """Выполнить задания последовательно или в пуле процессов."""
    job_list = list(jobs)
    if not job_list:
        return []

    n_workers = max(1, int(workers))
    if n_workers == 1 or len(job_list) == 1:
        return [render_wind_plot_job(job) for job in job_list]

    max_workers = min(n_workers, len(job_list))
    results: list[WindPlotRenderResult] = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(render_wind_plot_job, job) for job in job_list]
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda item: item.dataset_name)
