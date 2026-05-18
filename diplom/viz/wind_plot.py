"""3D-визуализация поля ветра ERA5 (Plotly HTML).

Строит интерактивный 3D-граф с конусами (Cone), отображающими направление и
скорость ветра на всех высотах, широтах и долготах для выбранного временного
среза ERA5-датасета.

Публичный API:
  list_available_times(path)               → list[np.datetime64]
  load_wind_slice(path, time_target)       → WindSlice
  build_wind_figure(slice_, **opts)        → go.Figure
  save_figure(fig, path)                   → standalone HTML
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
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



def _dipole_colorscale_default() -> list[list[float | str]]:
    """Тёмная сторона шкалы соответствует концам проекции → −1, светлая — +1 (на тёмном фоне)."""
    return [
        [0.0, "rgb(35,54,93)"],
        [0.5, "rgb(140,157,177)"],
        [1.0, "rgb(228,237,246)"],
    ]


def _dipole_proj_units(u: np.ndarray, v: np.ndarray, *, plane: str) -> np.ndarray:
    """Доля компоненты u или v в горизонтальной скорости; при почти штиле → 0."""
    uu = np.asarray(u, dtype=np.float64)
    vv = np.asarray(v, dtype=np.float64)
    sh = np.hypot(uu, vv)
    sh_safe = np.maximum(sh, 1e-9)
    p = plane.lower().strip()
    if p == "east":
        c = uu / sh_safe
    elif p == "north":
        c = vv / sh_safe
    else:
        raise ValueError(f"direction_plane должен быть 'east' или 'north', получено {plane!r}.")
    return np.clip(np.where(sh < 1e-10, 0.0, c), -1.0, 1.0)


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
    direction_plane: str = "east",
    cone_direction_colorscale: Optional[str | list[list[float | str]]] = None,
    cone_direction_reversescale: bool = False,
    cone_dir_amp_lo: float = 0.22,
    cone_dir_amp_hi: float = 1.0,
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
      • Цвет               → дивергентная шкала по проекции на выбранную ось безразмерным **u / |Vₕ|** или **v / |Vₕ|**
                             (минус конец шкалы — одно крайнее направление, плюс — противоположное). Шкала и подпись colorbar совпадают.
      • Размер (слабо)     → скорость (mag) умножается на множитель от этого же скаляра: Plotly раскрашивает только по ‖V‖,
                             поэтому длины конусов слегва меняются вместе с цветом; штиль — отдельно, серым.
      • Tooltip            → u, v, w (реальные), скорость, давление, высота, скаляр для шкалы.

    Args:
        slice_:       срез ERA5, полученный через `load_wind_slice`.
        title:        заголовок графика.
        stride_lon:   прореживание по долготе (1 = брать каждую точку).
        stride_lat:   прореживание по широте.
        stride_altitude_m: шаг сетки отрисовки по высоте в **метрах** (равномерно по оси Z).
                           Поля u, v, w линейно интерполируются между исходными уровнями ERA5 по высоте.
        w_scale:      масштаб вертикальной компоненты для наглядности.
        cone_sizeref: общий масштаб размера конусов Plotly; меньше значение = больше конусы.
        direction_plane: **east** (ось W→E, u / |Vₕ|) или **north** (ось S→N, v / |Vₕ|).
        cone_direction_colorscale: имя палитры Plotly («RdBu», «PuOr», …) или свой список [[0,color],[1,color],…];
                                  по умолчанию тёмная — светлая оттенки синего/серого.
        cone_direction_reversescale: перевернуть именованную палитру (игнорируется для пользовательского списка цветов).
        cone_dir_amp_lo / cone_dir_amp_hi: насколько множитель длины конусов меняется при скаляре −1 … +1 (0 < lo < hi).
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

    # ── Вектора для Cone: цвет = ‖V‖; подбираем длину так, чтобы ‖V‖ росло с дипольным скаляром ─────────
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
    plane_clean = direction_plane.strip().lower()
    dipole_flat = np.zeros(n_pts, dtype=np.float64)
    if np.any(active):
        dipole_flat[active] = _dipole_proj_units(u_arr[active], v_arr[active], plane=plane_clean)

    amp_lo = float(cone_dir_amp_lo)
    amp_hi = float(cone_dir_amp_hi)
    if not (0.0 < amp_lo < amp_hi):
        raise ValueError("Нужно 0 < cone_dir_amp_lo < cone_dir_amp_hi.")
    dir_amp = amp_lo + (amp_hi - amp_lo) * (dipole_flat + 1.0) * 0.5
    scale_amp = ref_speed * mag * dir_amp
    east = ux * scale_amp
    north = vy * scale_amp
    up = wz * scale_amp

    theta = np.arctan2(v_arr, u_arr)

    dipole_tickvals = [-1.0, 0.0, 1.0]
    if plane_clean == "north":
        cbar_title = "Направление (ось S→N)<br><sub>v / |V_h| −1 … +1 · юг → север</sub>"
        dipole_ticktext = ["юг", "0", "север"]
    else:
        cbar_title = "Направление (ось W→E)<br><sub>u / |V_h| −1 … +1 · запад → восток</sub>"
        dipole_ticktext = ["запад", "0", "восток"]

    cs_raw = cone_direction_colorscale if cone_direction_colorscale is not None else _dipole_colorscale_default()
    use_reverse = cone_direction_reversescale and isinstance(cs_raw, str)

    def hover_for_indices(idx: np.ndarray) -> list[str]:
        out: list[str] = []
        for i in idx:
            ii = int(i)
            spd = float(speed_flat[ii])
            compass_deg = np.degrees(theta[ii]) % 360.0
            dpl = float(dipole_flat[ii])
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
                f"<b>Шкала направления ≈ {dpl:+.2f}</b> (−1…+1)<br>"
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
        fig.add_trace(
            go.Cone(
                x=x_arr[idx_act].tolist(),
                y=y_arr[idx_act].tolist(),
                z=z_arr[idx_act].tolist(),
                u=east[idx_act].tolist(),
                v=north[idx_act].tolist(),
                w=up[idx_act].tolist(),
                colorscale=cs_raw,
                reversescale=use_reverse,
                sizemode="absolute",
                sizeref=float(cone_sizeref),
                anchor="tail",
                hovertemplate="%{text}<extra></extra>",
                text=hover_for_indices(idx_act),
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
                x=[x0, x0 + eps, x0 + 2 * eps],
                y=[y0, y0, y0],
                z=[z0, z0, z0],
                mode="markers",
                marker=dict(
                    color=dipole_tickvals,
                    cmin=-1,
                    cmax=1,
                    colorscale=cs_raw,
                    reversescale=use_reverse,
                    size=np.full(3, 2.2),
                    opacity=0,
                    showscale=True,
                    colorbar=dict(
                        title=dict(
                            text=cbar_title,
                            side="right",
                            font=dict(color="#e9ecf5", size=12),
                        ),
                        tickvals=dipole_tickvals,
                        ticktext=dipole_ticktext,
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
