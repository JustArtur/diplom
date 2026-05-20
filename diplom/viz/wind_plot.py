"""3D-визуализация поля ветра ERA5 (Plotly HTML).

Строит интерактивный 3D-граф с конусами (Cone), отображающими направление и
скорость ветра на всех высотах, широтах и долготах для выбранного временного
среза ERA5-датасета.

Публичный API:
  list_available_times(path)               → list[np.datetime64]
  resolve_wind_time(interpolator, target)  → np.datetime64
  compute_wind_steerability_stats(...)     → WindSteerabilityStats
  build_wind_figure(interpolator, time, **opts) → go.Figure
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

from diplom.geo import meters_per_deg_lat, meters_per_deg_lon
from diplom.wind.interp import WindInterpolator

# ──────────────────── Имена переменных ERA5 (list_available_times) ────────────────────

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


# Эталонные пары давления (гПа) для local directional shear: 700↔500, 500↔300, …
REFERENCE_SHEAR_PAIRS_HPA: tuple[tuple[float, float], ...] = (
    (700.0, 500.0),
    (500.0, 300.0),
    (300.0, 250.0),
    (250.0, 200.0),
    (200.0, 150.0),
)
DEFAULT_HEADING_BIN_DEG = 15.0
DEFAULT_STEERABILITY_WEIGHTS: tuple[float, float, float, float] = (0.30, 0.35, 0.15, 0.20)
DEFAULT_TEMPORAL_LAGS = (1, 3, 6)
_PAIR_PRESSURE_TOLERANCE_HPA = 75.0
_CURVATURE_NORM_DEG = 45.0


def _horizontal_vector_angle_deg(
    u1: np.ndarray,
    v1: np.ndarray,
    u2: np.ndarray,
    v2: np.ndarray,
    *,
    calm_speed_mps: float,
) -> np.ndarray:
    """Угол между горизонтальными векторами ветра через скалярное произведение, [0, 180]°."""
    dot = u1 * u2 + v1 * v2
    mag = np.hypot(u1, v1) * np.hypot(u2, v2)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_theta = np.clip(dot / mag, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_theta))
    calm_sq = float(calm_speed_mps) ** 2
    return np.where(mag >= calm_sq, angle, np.nan)


def _resolve_reference_pair_indices(
    pressure_hpa: np.ndarray,
    p_low_hpa: float,
    p_high_hpa: float,
    *,
    tolerance_hpa: float,
) -> tuple[int, int] | None:
    """Индексы уровней для пары (низкая высота ↔ высокая), или None если уровни недоступны."""
    idx_low = int(np.argmin(np.abs(pressure_hpa - p_low_hpa)))
    idx_high = int(np.argmin(np.abs(pressure_hpa - p_high_hpa)))
    if idx_low == idx_high:
        return None
    if abs(float(pressure_hpa[idx_low]) - p_low_hpa) > tolerance_hpa:
        return None
    if abs(float(pressure_hpa[idx_high]) - p_high_hpa) > tolerance_hpa:
        return None
    return idx_low, idx_high


def _unwrap_azimuth_along_levels(azimuth_deg: np.ndarray) -> np.ndarray:
    """Развернуть азимут вдоль оси уровней (axis=1) для расчёта кривизны."""
    step = azimuth_deg[:, 1:] - azimuth_deg[:, :-1]
    step = (step + 180.0) % 360.0 - 180.0
    return np.concatenate([azimuth_deg[:, :1], azimuth_deg[:, :1] + np.cumsum(step, axis=1)], axis=1)


def _weighted_steerability_score(
    d_local: float,
    heading_diversity: float,
    curvature_richness: float,
    temporal_persistence: float | None,
    *,
    weights: tuple[float, float, float, float],
) -> float:
    components = [d_local, heading_diversity, curvature_richness]
    active_weights = list(weights[:3])
    if temporal_persistence is not None:
        components.append(temporal_persistence)
        active_weights.append(weights[3])
    total = float(sum(active_weights))
    if total <= 0.0:
        return 0.0
    return float(sum(w * c for w, c in zip(active_weights, components)) / total)


@dataclass(frozen=True, slots=True)
class WindSteerabilityStats:
    """RL-ориентированные метрики управляемости ветром по всему датасету.

    Все компоненты нормализованы в [0, 1] и не зависят от размера сетки.
    """

    steerability_score: float
    d_local: float
    heading_diversity: float
    curvature_richness: float
    temporal_persistence: float | None
    d_local_angle_deg: float
    calm_speed_mps: float
    altitude_span_m: float
    n_pressure_levels: int
    n_time_steps: int
    weight_d_local: float
    weight_heading: float
    weight_curvature: float
    weight_temporal: float

    def summary_lines(self) -> tuple[str, ...]:
        """Краткие строки для лога и аннотации на графике."""
        lines = (
            f"Steerability Score: {100.0 * self.steerability_score:.1f}",
            f"D_local (сдвиг направления): {100.0 * self.d_local:.1f} "
            f"(θ={self.d_local_angle_deg:.1f}°)",
            f"H (разнообразие курсов): {100.0 * self.heading_diversity:.1f}",
            f"C (кривизна по высоте): {100.0 * self.curvature_richness:.1f}",
        )
        if self.temporal_persistence is not None:
            lines += (f"T (устойчивость во времени): {100.0 * self.temporal_persistence:.1f}",)
        else:
            lines += ("T (устойчивость во времени): n/a (один временной шаг)",)
        return lines


def compute_wind_steerability_stats(
    interpolator: WindInterpolator,
    *,
    calm_speed_mps: float = 0.12,
    heading_bin_deg: float = DEFAULT_HEADING_BIN_DEG,
    steerability_weights: tuple[float, float, float, float] = DEFAULT_STEERABILITY_WEIGHTS,
    temporal_lags: tuple[int, ...] = DEFAULT_TEMPORAL_LAGS,
) -> WindSteerabilityStats:
    """Оценить управляемость ветром для RL по всему ERA5-датасету.

    Компоненты (все [0, 1], size-invariant):

    * **D_local** — mean arccos(V_lo·V_hi / |V_lo||V_hi|) по эталонным парам
      700↔500, 500↔300, 300↔250, 250↔200, 200↔150 (доступным в файле).
    * **H** — доля покрытых 15°-секторов круга направлений по всем высотам в точке.
    * **C** — средняя «богатость» вертикальной кривизны: |d²θ/dz²| (развёрнутый азимут).
    * **T** — средняя векторная автокорреляция V(t) с V(t+τ), τ ∈ {1,3,6} шагов.

    **Steerability Score** = взвешенная сумма компонент (w = 0.30, 0.35, 0.15, 0.20).
    """
    if calm_speed_mps < 0.0:
        raise ValueError("calm_speed_mps должен быть ≥ 0.")
    if heading_bin_deg <= 0.0 or 360.0 % heading_bin_deg != 0.0:
        raise ValueError("heading_bin_deg должен делить 360.")
    if len(steerability_weights) != 4 or any(w < 0.0 for w in steerability_weights):
        raise ValueError("steerability_weights: четыре неотрицательных веса.")

    pressure_hpa = np.asarray(interpolator.pressure_axis_hpa, dtype=np.float64)
    alt_m = _pressure_to_altitude_m(pressure_hpa)
    level_order = np.argsort(alt_m)
    alt_sorted = alt_m[level_order]
    altitude_span_m = float(alt_sorted[-1] - alt_sorted[0]) if alt_sorted.size > 1 else 0.0

    u = np.asarray(interpolator.data[0], dtype=np.float64)[:, level_order, :, :]
    v = np.asarray(interpolator.data[1], dtype=np.float64)[:, level_order, :, :]
    speed = np.hypot(u, v)
    azimuth_deg = _horizontal_azimuth_deg_east_north(u, v)

    n_time = int(u.shape[0])
    n_levels = int(u.shape[1])
    valid = speed >= calm_speed_mps

    # ── D_local: эталонные пары давления ─────────────────────────────────
    pair_angles: list[np.ndarray] = []
    for p_low, p_high in REFERENCE_SHEAR_PAIRS_HPA:
        resolved = _resolve_reference_pair_indices(
            pressure_hpa,
            p_low,
            p_high,
            tolerance_hpa=_PAIR_PRESSURE_TOLERANCE_HPA,
        )
        if resolved is None:
            continue
        idx_native_low, idx_native_high = resolved
        idx_sorted_low = int(np.where(level_order == idx_native_low)[0][0])
        idx_sorted_high = int(np.where(level_order == idx_native_high)[0][0])
        angle = _horizontal_vector_angle_deg(
            u[:, idx_sorted_low],
            v[:, idx_sorted_low],
            u[:, idx_sorted_high],
            v[:, idx_sorted_high],
            calm_speed_mps=calm_speed_mps,
        )
        pair_angles.append(angle.ravel())

    if pair_angles:
        all_pair_angles = np.concatenate(pair_angles)
        d_local_angle_deg = float(np.nanmean(all_pair_angles))
        d_local = float(np.clip(d_local_angle_deg / 180.0, 0.0, 1.0))
    else:
        d_local_angle_deg = 0.0
        d_local = 0.0

    # ── H: reachable heading diversity (coverage по высотам) ───────────────
    n_bins = int(360.0 / heading_bin_deg)
    bin_idx = (np.floor(azimuth_deg / heading_bin_deg).astype(np.intp) % n_bins)
    n_spatial = int(valid.shape[2] * valid.shape[3])
    covered = np.zeros((n_time, n_spatial, n_bins), dtype=np.bool_)
    valid_flat = valid.reshape(n_time, n_levels, n_spatial)
    bins_flat = bin_idx.reshape(n_time, n_levels, n_spatial)
    for level in range(n_levels):
        level_valid = valid_flat[:, level, :]
        level_bins = bins_flat[:, level, :]
        t_idx, s_idx = np.nonzero(level_valid)
        covered[t_idx, s_idx, level_bins[t_idx, s_idx]] = True

    min_valid_levels = 2
    level_counts = valid_flat.sum(axis=1)
    diversity_mask = level_counts >= min_valid_levels
    heading_diversity = (
        float(np.mean(covered[diversity_mask].sum(axis=-1) / n_bins))
        if np.any(diversity_mask)
        else 0.0
    )

    # ── C: vertical wind curvature |d²θ/dz²| ─────────────────────────────
    if n_levels >= 3 and alt_sorted.size >= 3:
        theta_unwrapped = _unwrap_azimuth_along_levels(azimuth_deg)
        dz = np.diff(alt_sorted)
        mean_dz = float(np.mean(dz))
        d2_theta = (
            theta_unwrapped[:, 2:]
            - 2.0 * theta_unwrapped[:, 1:-1]
            + theta_unwrapped[:, :-2]
        )
        if mean_dz > 1e-6:
            curvature = np.abs(d2_theta) / (mean_dz * mean_dz)
        else:
            curvature = np.abs(d2_theta)
        valid_triple = valid[:, :-2] & valid[:, 1:-1] & valid[:, 2:]
        curvature_richness = (
            float(np.clip(np.nanmean(curvature[valid_triple]) / _CURVATURE_NORM_DEG, 0.0, 1.0))
            if np.any(valid_triple)
            else 0.0
        )
    else:
        curvature_richness = 0.0

    # ── T: temporal persistence (vector autocorrelation) ───────────────────
    temporal_persistence: float | None
    if n_time >= 2:
        lag_scores: list[float] = []
        for lag in temporal_lags:
            if lag >= n_time:
                continue
            u0 = u[:-lag]
            v0 = v[:-lag]
            u1 = u[lag:]
            v1 = v[lag:]
            dot = u0 * u1 + v0 * v1
            mag = np.hypot(u0, v0) * np.hypot(u1, v1)
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = np.clip(dot / mag, -1.0, 1.0)
            lag_valid = (np.hypot(u0, v0) >= calm_speed_mps) & (np.hypot(u1, v1) >= calm_speed_mps)
            if np.any(lag_valid):
                lag_scores.append(float((np.nanmean(corr[lag_valid]) + 1.0) * 0.5))
        temporal_persistence = float(np.mean(lag_scores)) if lag_scores else None
    else:
        temporal_persistence = None

    steerability_score = _weighted_steerability_score(
        d_local,
        heading_diversity,
        curvature_richness,
        temporal_persistence,
        weights=steerability_weights,
    )

    w_d, w_h, w_c, w_t = steerability_weights
    return WindSteerabilityStats(
        steerability_score=steerability_score,
        d_local=d_local,
        heading_diversity=heading_diversity,
        curvature_richness=curvature_richness,
        temporal_persistence=temporal_persistence,
        d_local_angle_deg=d_local_angle_deg,
        calm_speed_mps=float(calm_speed_mps),
        altitude_span_m=altitude_span_m,
        n_pressure_levels=n_levels,
        n_time_steps=n_time,
        weight_d_local=float(w_d),
        weight_heading=float(w_h),
        weight_curvature=float(w_c),
        weight_temporal=float(w_t),
    )


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


def _sample_wind_on_grid(
    interpolator: WindInterpolator,
    time: np.datetime64,
    *,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    altitude_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Сэмплировать u, v, w и давление на 3D-сетке через ``WindInterpolator.batch_vector_at``."""
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


# ──────────────────── Список доступных временных меток ────────────────────


def list_available_times(path: Path) -> List[np.datetime64]:
    """Вернуть список временных меток, доступных в ERA5-файле."""
    with xr.open_dataset(path) as ds:
        times = ds[_TIME_DIM].values.astype("datetime64[s]")
    return [np.datetime64(t, "s") for t in times]


def resolve_wind_time(interpolator: WindInterpolator, time_target: np.datetime64) -> np.datetime64:
    """Выбрать ближайший к ``time_target`` шаг из оси времени интерполятора."""
    times_ns = interpolator.time_axis_ns.astype("datetime64[ns]")
    target_ns = np.datetime64(time_target, "ns")
    time_idx = int(np.argmin(np.abs(times_ns.astype(np.int64) - target_ns.astype(np.int64))))
    return np.datetime64(int(interpolator.time_axis_ns[time_idx]), "ns").astype("datetime64[s]")


# ──────────────────── Построение 3D-графика ────────────────────


def build_wind_figure(
    interpolator: WindInterpolator,
    time: np.datetime64,
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
    steerability_stats: Optional[WindSteerabilityStats] = None,
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
        interpolator: интерполятор ERA5 (тот же, что в обучении и симуляции).
        time:         временна́я метка среза; клампится к диапазону датасета.
        title:        заголовок графика.
        stride_lon:   прореживание по долготе (1 = брать каждую точку).
        stride_lat:   прореживание по широте.
        stride_altitude_m: шаг сетки отрисовки по высоте в **метрах** (равномерно по оси Z).
                           u, v, w берутся через ``WindInterpolator.batch_vector_at`` (4D-интерполяция).
        w_scale:      масштаб вертикальной компоненты для наглядности.
        cone_sizeref: общий масштаб размера конусов Plotly; меньше значение = больше конусы.
        cone_azimuth_colorscale: циклическая палитра [[0,color],…,[1,color]] или имя Plotly; по умолчанию HSL-колесо.
                                   Для азимута нужна палитра, где первый и последний цвет совпадают.
        cone_azimuth_reversescale: перевернуть именованную палитру (игнорируется для пользовательского списка).
        cone_speed_floor: нижняя граница множителя от нормализованной скорости (узкий разброс длин между точками).
        cone_speed_power: степень нормализованной горизонтальной скорости для этого множителя.
        calm_speed_mps: |V_h| ниже порога → серый отдельный trace (штиль).
        steerability_stats: готовые RL-метрики управляемости; если None — считаются по всему датасету.
        altitude_unit: «km» или «m» — единицы на оси Z.
        origin_lat/origin_lon: опорная точка для локальных метровых координат (границы сцены как у `WindInterpolator.world_bounds`).
    """
    plot_time = np.datetime64(time, "ns")
    lat_axis = np.asarray(interpolator.latitude_axis_deg, dtype=np.float64)
    lon_axis = np.asarray(interpolator.longitude_axis_deg, dtype=np.float64)
    altitude_axis = _pressure_to_altitude_m(interpolator.pressure_axis_hpa).astype(np.float64)

    # ── Равномерная сетка по высоте (м), ветер через batch_vector_at ──
    h_targets = _altitude_targets_m(altitude_axis, float(stride_altitude_m))
    alt_sub = h_targets.astype(np.float32)

    # ── Прореживание по горизонтали ──────────────────────────────────────
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

    world_bounds_xy = interpolator.world_bounds
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
        else f"Поле ветра ERA5 · {np.datetime64(plot_time, 's')}"
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
    if steerability_stats is None:
        steerability_stats = compute_wind_steerability_stats(
            interpolator,
            calm_speed_mps=float(calm_speed_mps),
        )

    annotations: list[dict] = [
        dict(
            text=(
                "<b>Steerability</b> (весь датасет)<br>"
                + "<br>".join(steerability_stats.summary_lines())
                + f"<br><span style='color:rgba(180,190,210,0.85)'>"
                f"Уровней: {steerability_stats.n_pressure_levels} · "
                f"T={steerability_stats.n_time_steps} · "
                f"Δh: {steerability_stats.altitude_span_m / 1000.0:.1f} км · "
                f"|V_h| ≥ {steerability_stats.calm_speed_mps:.2f} м/с</span>"
            ),
            xref="paper",
            yref="paper",
            x=0.01,
            y=0.99,
            xanchor="left",
            yanchor="top",
            showarrow=False,
            align="left",
            bgcolor="rgba(22,26,42,0.88)",
            bordercolor="rgba(120,130,160,0.55)",
            borderwidth=1,
            borderpad=8,
            font=dict(size=11, color="#e9ecf5"),
        )
    ]
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
    force: bool = False


@dataclass(frozen=True, slots=True)
class WindPlotRenderResult:
    dataset_name: str
    plot_path: Path | None
    saved: bool
    log_lines: tuple[str, ...]
    steerability_stats: WindSteerabilityStats | None = None
    error: str | None = None


def render_wind_plot_job(job: WindPlotRenderJob) -> WindPlotRenderResult:
    """Построить и сохранить график для одного датасета (отдельный процесс)."""
    from diplom.data.era5_paths import era5_dataset_title, wind_plot_html_path

    name = job.dataset_path.name
    plot_path = wind_plot_html_path(job.dataset_path, job.output_dir)
    logs: list[str] = []

    if plot_path.exists() and not job.force:
        return WindPlotRenderResult(
            dataset_name=name,
            plot_path=plot_path,
            saved=False,
            log_lines=(f"Пропуск {name}: график уже есть → {plot_path}",),
        )
    if plot_path.exists() and job.force:
        logs.append(f"Пересоздаю {name}: --force → {plot_path}")

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

        logs.append(f"Загружаю интерполятор ERA5: {job.dataset_path} @ {target_time} …")
        interpolator = WindInterpolator.from_file(job.dataset_path)
        try:
            plot_time = resolve_wind_time(interpolator, target_time)
            logs.append(
                f"Интерполятор готов · {name} · время={plot_time} "
                f"· уровней={len(interpolator.pressure_axis_hpa)} "
                f"· lat={len(interpolator.latitude_axis_deg)} "
                f"· lon={len(interpolator.longitude_axis_deg)}"
            )

            wb = interpolator.world_bounds
            logs.append(
                f"[wind-viz] {name} · X [{wb.x_min:.1f} … {wb.x_max:.1f}] м · "
                f"Y [{wb.y_min:.1f} … {wb.y_max:.1f}] м · "
                f"Z [{wb.z_min:.1f} … {wb.z_max:.1f}] м"
            )

            steerability_stats = compute_wind_steerability_stats(interpolator)
            for line in steerability_stats.summary_lines():
                logs.append(f"[steerability] {name} · {line}")

            fig = build_wind_figure(
                interpolator,
                plot_time,
                title=era5_dataset_title(job.dataset_path),
                stride_lon=job.stride_lon,
                stride_lat=job.stride_lat,
                stride_altitude_m=job.stride_altitude_m,
                w_scale=job.w_scale,
                steerability_stats=steerability_stats,
            )
        finally:
            interpolator.close()
        save_figure(fig, plot_path)
        logs.append(f"График сохранён: {plot_path}")
        return WindPlotRenderResult(
            dataset_name=name,
            plot_path=plot_path,
            saved=True,
            log_lines=tuple(logs),
            steerability_stats=steerability_stats,
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
