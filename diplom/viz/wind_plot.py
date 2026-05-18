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

from diplom.geo import meters_per_deg_lat, meters_per_deg_lon, pressure_hpa_to_altitude_m
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


def _ensure_scene_axis_range(lo: float, hi: float, *, min_half_width_m: float = 500.0) -> tuple[float, float]:
    """Plotly не любит совпадающие min/max; для решётки из одной точки расширяем полуширину."""
    center = (lo + hi) * 0.5
    half = max((hi - lo) * 0.5, min_half_width_m)
    return center - half, center + half


def _pressure_targets_pa(p_hpa: np.ndarray, step_pa: float) -> np.ndarray:
    """Равномерная сетка давления в паскалях от минимума к максимуму в пределах среза.

    ``step_pa`` — расстояние между соседними уровнями для отрисовки (как в CLI ``stride_level``).
    """
    if step_pa <= 0:
        raise ValueError("stride_level (шаг по давлению, Па) должен быть > 0.")
    p_pa = np.sort(np.unique(np.asarray(p_hpa, dtype=np.float64) * 100.0))
    if p_pa.size == 0:
        raise ValueError("Датасет не содержит уровней давления.")
    lo, hi = float(p_pa[0]), float(p_pa[-1])
    if hi - lo < 1e-6:
        return np.array([lo], dtype=np.float64)
    pts = np.arange(lo, hi, step_pa, dtype=np.float64)
    if pts.size == 0 or abs(float(pts[-1]) - hi) > max(1e-6, 1e-3 * step_pa):
        pts = np.append(pts, hi)
    else:
        pts[-1] = hi
    return pts


def _interp_wind_vertical_to_pressure(
    slice_: WindSlice,
    p_targets_pa: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Линейная интерполяция u, v, w между уровнями ERA5 на заданные давления (Па)."""
    order = np.argsort(slice_.pressure)
    p_asc = slice_.pressure[order].astype(np.float64) * 100.0
    u_asc = slice_.u[order, ...].astype(np.float64)
    v_asc = slice_.v[order, ...].astype(np.float64)
    w_asc = slice_.w[order, ...].astype(np.float64)
    if p_asc.size == 1:
        shp = (len(p_targets_pa),) + tuple(u_asc.shape[1:])
        return (
            np.broadcast_to(u_asc[0], shp).astype(np.float32).copy(),
            np.broadcast_to(v_asc[0], shp).astype(np.float32).copy(),
            np.broadcast_to(w_asc[0], shp).astype(np.float32).copy(),
        )
    pt = np.clip(p_targets_pa.astype(np.float64), p_asc[0], p_asc[-1])
    kw = dict(axis=0, kind="linear", bounds_error=False, fill_value=(u_asc[0], u_asc[-1]))
    u_i = interp1d(p_asc, u_asc, **kw)(pt).astype(np.float32)
    v_i = interp1d(p_asc, v_asc, **kw)(pt).astype(np.float32)
    w_i = interp1d(p_asc, w_asc, **kw)(pt).astype(np.float32)
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
    stride_level: float = 5000.0,
    w_scale: float = 0.0,
    cone_sizeref: float = 50,
    colorscale: str = "Plasma",
    altitude_unit: str = "km",
    origin_lat: Optional[float] = None,
    origin_lon: Optional[float] = None,
) -> go.Figure:
    """Построить интерактивный 3D-граф поля ветра ERA5.

    Конусы (Cone) кодируют:
      • Направление острия  → направление ветра (u, v, w·w_scale).
      • Цвет               → горизонтальная скорость ветра (м/с).
      • Tooltip            → u, v, w (реальные), скорость, давление, высота.

    Args:
        slice_:       срез ERA5, полученный через `load_wind_slice`.
        title:        заголовок графика.
        stride_lon:   прореживание по долготе (1 = брать каждую точку).
        stride_lat:   прореживание по широте.
        stride_level: шаг сетки отрисовки по давлению в **паскалях** (напр. 5000 ≈ каждые 50 гПа).
                      Поля u, v, w линейно интерполируются между исходными уровнями ERA5.
        w_scale:      масштаб вертикальной компоненты для наглядности.
        cone_sizeref: общий масштаб размера конусов Plotly; меньше значение = больше конусы.
        colorscale:   Plotly colorscale для скорости ветра.
        altitude_unit: «km» или «m» — единицы на оси Z.
        origin_lat/origin_lon: опорная точка для локальных метровых координат.
          Оси X/Y/Z задаются жёстко по границам датасета (совпадает с `WindInterpolator.world_bounds`).
    """
    # ── Вертикаль: интерполяция на сетку с шагом stride_level (Па) ──────
    p_targets_pa = _pressure_targets_pa(slice_.pressure, float(stride_level))
    u_full, v_full, w_full = _interp_wind_vertical_to_pressure(slice_, p_targets_pa)
    pressure_sub_hpa = (p_targets_pa / 100.0).astype(np.float32)
    alt_sub = pressure_hpa_to_altitude_m(pressure_sub_hpa.astype(np.float64)).astype(np.float32)
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

    # ── Flatten ──────────────────────────────────────────────────────────
    x_flat = x_3d.ravel().tolist()
    y_flat = y_3d.ravel().tolist()
    z_flat = z_values.ravel().tolist()
    u_flat = u_sub.ravel().tolist()
    v_flat = v_sub.ravel().tolist()
    # По умолчанию вертикаль не участвует в ориентации конуса, чтобы не "заваливать" стрелки вверх/вниз.
    w_vis = (w_sub * w_scale).ravel().tolist()
    # Реальные значения w для tooltip
    w_real = w_sub.ravel()
    speed_flat = speed_sub.ravel()
    p_flat = p_3d.ravel()
    alt_flat = alt_3d.ravel()

    hover_texts = [
        (
            f"<b>x={x_flat[i]:.0f} м, y={y_flat[i]:.0f} м</b><br>"
            f"Опорная точка: lon={origin_lon:.2f}°, lat={origin_lat:.2f}°<br>"
            f"Давление: {p_flat[i]:.2f} гПа<br>"
            f"Высота: {alt_flat[i]/1000:.1f} км ({alt_flat[i]:.0f} м)<br>"
            f"<b>u = {u_flat[i]:.2f} м/с</b> (W→E)<br>"
            f"<b>v = {v_flat[i]:.2f} м/с</b> (S→N)<br>"
            f"<b>w = {w_real[i]:.4f} м/с</b> (↑+)<br>"
            f"<b>|V_h| = {speed_flat[i]:.2f} м/с</b>"
        )
        for i in range(len(x_flat))
    ]

    z_label = "Высота, км" if altitude_unit == "km" else "Высота, м"
    x_label = "X, м"
    y_label = "Y, м"
    auto_title = (
        title
        if title is not None
        else f"Поле ветра ERA5 · {slice_.time}"
    )

    fig = go.Figure()

    fig.add_trace(
        go.Cone(
            x=x_flat,
            y=y_flat,
            z=z_flat,
            u=u_flat,
            v=v_flat,
            w=w_vis,
            colorscale=colorscale,
            cmin=float(speed_flat.min()),
            cmax=float(speed_flat.max()),
            colorbar=dict(
                title=dict(text="Гор. скорость, м/с", side="right"),
                thickness=15,
                len=0.7,
            ),
            sizemode="absolute",
            sizeref=cone_sizeref,
            anchor="tail",
            hovertemplate="%{text}<extra></extra>",
            text=hover_texts,
            name="Ветер",
            showscale=True,
        )
    )

    # ── Аннотация о масштабировании w ────────────────────────────────────
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
        margin=dict(l=0, r=0, b=30, t=60),
        template="plotly_dark",
        paper_bgcolor="rgba(15,15,25,1)",
    )

    return fig


# ──────────────────── Сохранение ────────────────────


def save_figure(fig: go.Figure, path: Path) -> None:
    """Сохранить фигуру как standalone HTML (Plotly CDN, без встроенного JS)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")
