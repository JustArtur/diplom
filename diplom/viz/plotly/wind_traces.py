"""Plotly-traces поля ветра (конусы) для переиспользования в wind-viz и trajectory overlay."""

from __future__ import annotations

import colorsys
import re
from typing import Literal, Optional

import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go

from diplom.geo import meters_per_deg_lat, meters_per_deg_lon
from diplom.wind.interp import WindInterpolator

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

CONE_AZIMUTH_BINS = 36
_CALM_RGB = "rgb(138,143,156)"


def horizontal_azimuth_deg_east_north(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Горизонтальный азимут ветра: 0° = E, 90° = N, диапазон [0, 360)."""
    return np.mod(
        np.degrees(np.arctan2(np.asarray(v, dtype=np.float64), np.asarray(u, dtype=np.float64))),
        360.0,
    )


def pressure_to_altitude_m(pressure_hpa: np.ndarray) -> np.ndarray:
    """Обратная барометрическая формула ISA: давление (гПа) → высота (м)."""
    p0 = 1013.25
    t0 = 288.15
    lapse = 0.0065
    g = 9.80665
    r = 8.31447
    m = 0.0289644
    exp = (g * m) / (r * lapse)
    ratio = np.clip(np.asarray(pressure_hpa, dtype=np.float64) / p0, 1e-9, 1.0)
    return (t0 / lapse) * (1.0 - np.power(ratio, 1.0 / exp))


def build_wind_cone_traces(
    interpolator: WindInterpolator,
    time: np.datetime64,
    *,
    stride_lon: int = 1,
    stride_lat: int = 1,
    stride_altitude_m: float = 500.0,
    w_scale: float = 0.0,
    cone_sizeref: float = 20.0,
    cone_azimuth_colorscale: Optional[str | list[list[float | str]]] = None,
    cone_azimuth_reversescale: bool = False,
    cone_speed_floor: float = 0.38,
    cone_speed_power: float = 0.42,
    calm_speed_mps: float = 0.12,
    altitude_unit: Literal["m", "km"] = "m",
    origin_lat: Optional[float] = None,
    origin_lon: Optional[float] = None,
    show_colorbar: bool = False,
    show_calm: bool = True,
    cone_opacity: float = 1.0,
    cone_scale: float = 1.0,
) -> list[go.BaseTraceType]:
    """Построить Plotly Cone-traces поля ветра на 3D-сетке ERA5."""
    plot_time = np.datetime64(time, "ns")
    lat_axis = np.asarray(interpolator.latitude_axis_deg, dtype=np.float64)
    lon_axis = np.asarray(interpolator.longitude_axis_deg, dtype=np.float64)
    altitude_axis = pressure_to_altitude_m(interpolator.pressure_axis_hpa).astype(np.float64)

    h_targets = _altitude_targets_m(altitude_axis, float(stride_altitude_m))
    alt_sub = h_targets.astype(np.float32)

    la_idx = np.arange(0, len(lat_axis), max(1, stride_lat))
    lo_idx = np.arange(0, len(lon_axis), max(1, stride_lon))
    lat_sub = lat_axis[la_idx]
    lon_sub = lon_axis[lo_idx]

    u_sub, v_sub, w_sub, pressure_sub = _sample_wind_on_grid(
        interpolator,
        plot_time,
        lat_deg=lat_sub,
        lon_deg=lon_sub,
        altitude_m=h_targets,
    )
    speed_sub = np.sqrt(u_sub**2 + v_sub**2).astype(np.float32)

    origin_lat = float(origin_lat) if origin_lat is not None else float(interpolator.origin_lat)
    origin_lon = float(origin_lon) if origin_lon is not None else float(interpolator.origin_lon)

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
    ref_speed = float(np.max(speed_flat)) if x_arr.size > 0 else 1.0
    ref_speed = max(ref_speed, float(calm_speed_mps) * 2.0, 1.0)

    calm = speed_flat < float(calm_speed_mps)
    active = ~calm
    azimuth_deg = horizontal_azimuth_deg_east_north(u_arr, v_arr)

    scale_amp = ref_speed * mag
    east = ux * scale_amp
    north = vy * scale_amp
    up = wz * scale_amp

    cone_sizeref_effective = float(cone_sizeref) * float(cone_scale)

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

    traces: list[go.BaseTraceType] = []

    idx_calm = np.flatnonzero(calm)
    if show_calm and idx_calm.size > 0:
        flat_cs = [[0.0, _CALM_RGB], [1.0, _CALM_RGB]]
        traces.append(
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
                sizeref=cone_sizeref_effective,
                anchor="tail",
                opacity=float(cone_opacity),
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
            traces.append(
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
                    sizeref=cone_sizeref_effective,
                    anchor="tail",
                    opacity=float(cone_opacity),
                    hovertemplate="%{text}<extra></extra>",
                    text=hover_for_indices(sel),
                    showlegend=False,
                    showscale=False,
                )
            )

        if show_colorbar:
            wb = interpolator.world_bounds
            x0, y0, z0 = float(wb.x_min), float(wb.y_min), float(wb.z_min)
            if altitude_unit == "km":
                z0 /= 1000.0
            eps = max(wb.width, wb.height, wb.z_max - wb.z_min, 1.0) * 1e-6
            azimuth_tickvals = [0.0, 90.0, 180.0, 270.0]
            azimuth_ticktext = ["E (0°)", "N (90°)", "W (180°)", "S (270°)"]
            traces.append(
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
                                text=(
                                    "Направление ветра (азимут)<br>"
                                    "<sub>0° = E, 90° = N · горизонтальное</sub>"
                                ),
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

    return traces


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
    rgb = _sample_azimuth_color(colorscale, azimuth_deg, reversescale=reversescale)
    return [[0.0, rgb], [1.0, rgb]]


def _compressed_speed_mag(
    speed: np.ndarray,
    *,
    floor: float,
    power: float,
) -> np.ndarray:
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


def _altitude_targets_m(h_m: np.ndarray, step_m: float) -> np.ndarray:
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


def _sample_wind_on_grid(
    interpolator: WindInterpolator,
    time: np.datetime64,
    *,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    altitude_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    alt_3d, lat_3d, lon_3d = np.meshgrid(
        altitude_m.astype(np.float64),
        lat_deg.astype(np.float64),
        lon_deg.astype(np.float64),
        indexing="ij",
    )
    x_3d, y_3d = _lonlat_to_local_meters(
        lon_3d,
        lat_3d,
        origin_lon=interpolator.origin_lon,
        origin_lat=interpolator.origin_lat,
    )
    x_flat = x_3d.ravel()
    y_flat = y_3d.ravel()
    z_flat = alt_3d.ravel()
    n_pts = x_flat.size
    time_flat = np.full(n_pts, np.datetime64(time, "ns"), dtype="datetime64[ns]")

    wind = interpolator.batch_vector_at(x_flat, y_flat, z_flat, time_flat)
    grid_shape = alt_3d.shape
    u = wind[:, 0].reshape(grid_shape).astype(np.float32)
    v = wind[:, 1].reshape(grid_shape).astype(np.float32)
    w = wind[:, 2].reshape(grid_shape).astype(np.float32)
    pressure = interpolator._z_to_pressure(z_flat).reshape(grid_shape).astype(np.float32)
    return u, v, w, pressure


def _lonlat_to_local_meters(
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    origin_lon: float,
    origin_lat: float,
) -> tuple[np.ndarray, np.ndarray]:
    m_per_lat = meters_per_deg_lat(origin_lat)
    m_per_lon = meters_per_deg_lon(origin_lat)
    x_m = (np.asarray(lon, dtype=np.float64) - origin_lon) * m_per_lon
    y_m = (np.asarray(lat, dtype=np.float64) - origin_lat) * m_per_lat
    return x_m.astype(np.float32), y_m.astype(np.float32)
