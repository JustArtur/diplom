# 3D-визуализация поля ветра ERA5 (Plotly HTML).
#
# Строит интерактивный 3D-граф с конусами (Cone), отображающими направление и
# скорость ветра на всех высотах, широтах и долготах для выбранного временного
# среза ERA5-датасета.
#
# Публичный API:
# list_available_times(path)               -> list[np.datetime64]
# resolve_wind_time(interpolator, target)  -> np.datetime64
# compute_wind_steerability_stats(...)     -> WindSteerabilityStats
# build_wind_figure(interpolator, time, **opts) -> go.Figure
# save_figure(fig, path)                   -> standalone HTML
#
# Пакетная отрисовка датасетов из data/preview/, команда CLI diplom wind-viz.

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Sequence

import numpy as np
import plotly.graph_objects as go
import xarray as xr

from diplom.viz.plotly.wind_traces import (
    build_wind_cone_traces,
    horizontal_azimuth_deg_east_north,
    pressure_to_altitude_m,
)
from diplom.wind.interp import WindInterpolator

# Имена переменных ERA5 (list_available_times)
_TIME_DIM = "valid_time"

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
    # Угол между горизонтальными векторами ветра через скалярное произведение, [0, 180]°.
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
    # Индексы уровней для пары (низкая высота / высокая), или None если уровни недоступны.
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
    # Развернуть азимут вдоль оси уровней (axis=1) для расчёта кривизны.
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
    # RL-ориентированные метрики управляемости ветром по всему датасету.
    #
    # Все компоненты нормализованы в [0, 1] и не зависят от размера сетки.
    #

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
        # Краткие строки для лога и аннотации на графике.
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
    # Оценить управляемость ветром для RL по всему ERA5-датасету.
    #
    # Компоненты (все [0, 1], size-invariant):
    #
    # * **D_local**, mean arccos(V_lo·V_hi / |V_lo||V_hi|) по эталонным парам
    # 700↔500, 500↔300, 300↔250, 250↔200, 200↔150 (доступным в файле).
    # * **H**, доля покрытых 15°-секторов круга направлений по всем высотам в точке.
    # * **C**, средняя богатость вертикальной кривизны: |d²θ/dz²| (развёрнутый азимут).
    # * **T**, средняя векторная автокорреляция V(t) с V(t+τ), τ ∈ {1,3,6} шагов.
    #
    # **Steerability Score** = взвешенная сумма компонент (w = 0.30, 0.35, 0.15, 0.20).
    #
    if calm_speed_mps < 0.0:
        raise ValueError("calm_speed_mps должен быть ≥ 0.")
    if heading_bin_deg <= 0.0 or 360.0 % heading_bin_deg != 0.0:
        raise ValueError("heading_bin_deg должен делить 360.")
    if len(steerability_weights) != 4 or any(w < 0.0 for w in steerability_weights):
        raise ValueError("steerability_weights: четыре неотрицательных веса.")

    pressure_hpa = np.asarray(interpolator.pressure_axis_hpa, dtype=np.float64)
    alt_m = pressure_to_altitude_m(pressure_hpa)
    level_order = np.argsort(alt_m)
    alt_sorted = alt_m[level_order]
    altitude_span_m = float(alt_sorted[-1] - alt_sorted[0]) if alt_sorted.size > 1 else 0.0

    u = np.asarray(interpolator.data[0], dtype=np.float64)[:, level_order, :, :]
    v = np.asarray(interpolator.data[1], dtype=np.float64)[:, level_order, :, :]
    speed = np.hypot(u, v)
    azimuth_deg = horizontal_azimuth_deg_east_north(u, v)

    n_time = int(u.shape[0])
    n_levels = int(u.shape[1])
    valid = speed >= calm_speed_mps

    # D_local: эталонные пары давления
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

    # H: reachable heading diversity (coverage по высотам)
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

    # C: vertical wind curvature |d²θ/dz²|
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

    # T: temporal persistence (vector autocorrelation)
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


def _ensure_scene_axis_range(lo: float, hi: float, *, min_half_width_m: float = 500.0) -> tuple[float, float]:
    # Plotly не любит совпадающие min/max; для решётки из одной точки расширяем полуширину.
    center = (lo + hi) * 0.5
    half = max((hi - lo) * 0.5, min_half_width_m)
    return center - half, center + half


# Список доступных временных меток
def list_available_times(path: Path) -> List[np.datetime64]:
    # Вернуть список временных меток, доступных в ERA5-файле.
    with xr.open_dataset(path) as ds:
        times = ds[_TIME_DIM].values.astype("datetime64[s]")
    return [np.datetime64(t, "s") for t in times]


def resolve_wind_time(interpolator: WindInterpolator, time_target: np.datetime64) -> np.datetime64:
    # Выбрать ближайший к time_target шаг из оси времени интерполятора.
    times_ns = interpolator.time_axis_ns.astype("datetime64[ns]")
    target_ns = np.datetime64(time_target, "ns")
    time_idx = int(np.argmin(np.abs(times_ns.astype(np.int64) - target_ns.astype(np.int64))))
    return np.datetime64(int(interpolator.time_axis_ns[time_idx]), "ns").astype("datetime64[s]")


# Построение 3D-графика
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
    # Построить интерактивный 3D-граф поля ветра ERA5.
    #
    # Конусы (Cone) кодируют:
    # - Направление острия  -> направление ветра (u, v, w·w_scale).
    # - Цвет               -> горизонтальный азимут **arctan2(v, u)** в градусах (0° = E, 90° = N) с
    # **циклической** палитрой: близкие направления (напр. 350° и 30°) дают близкий цвет,
    # противоположные (≈180°), максимально различный.
    # - Размер (слабо)     -> горизонтальная скорость через сжатый множитель cone_speed_*; штиль, отдельно, серым.
    # - Tooltip            -> u, v, w (реальные), скорость, давление, высота, азимут.
    #
    # interpolator, тот же ERA5-интерполятор, что в обучении. time клампится к датасету.
    # stride_lon/lat, прореживание сетки; stride_altitude_m, шаг по высоте (м), u/v/w через
    # WindInterpolator.batch_vector_at. cone_sizeref и cone_speed_* задают размер конусов;
    # calm_speed_mps, порог штиля (отдельный серый trace). steerability_stats можно передать
    # готовыми; иначе считаются по датасету. altitude_unit: km или m; origin_lat/lon, опорная точка сцены.
    #
    plot_time = np.datetime64(time, "ns")
    altitude_unit_norm: Literal["m", "km"] = "km" if altitude_unit == "km" else "m"

    world_bounds_xy = interpolator.world_bounds
    x_lo, x_hi = _ensure_scene_axis_range(world_bounds_xy.x_min, world_bounds_xy.x_max)
    y_lo, y_hi = _ensure_scene_axis_range(world_bounds_xy.y_min, world_bounds_xy.y_max)
    z_lo_m, z_hi_m = _ensure_scene_axis_range(world_bounds_xy.z_min, world_bounds_xy.z_max, min_half_width_m=200.0)
    x_scene_range = [x_lo, x_hi]
    y_scene_range = [y_lo, y_hi]
    z_scene_range = (
        [z_lo_m / 1000.0, z_hi_m / 1000.0] if altitude_unit_norm == "km" else [z_lo_m, z_hi_m]
    )

    z_label = "Высота, км" if altitude_unit_norm == "km" else "Высота, м"
    auto_title = (
        title
        if title is not None
        else f"Поле ветра ERA5 · {np.datetime64(plot_time, 's')}"
    )

    fig = go.Figure()
    for trace in build_wind_cone_traces(
        interpolator,
        plot_time,
        stride_lon=stride_lon,
        stride_lat=stride_lat,
        stride_altitude_m=stride_altitude_m,
        w_scale=w_scale,
        cone_sizeref=cone_sizeref,
        cone_azimuth_colorscale=cone_azimuth_colorscale,
        cone_azimuth_reversescale=cone_azimuth_reversescale,
        cone_speed_floor=cone_speed_floor,
        cone_speed_power=cone_speed_power,
        calm_speed_mps=calm_speed_mps,
        altitude_unit=altitude_unit_norm,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        show_colorbar=True,
    ):
        fig.add_trace(trace)

    # Аннотации
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
            xaxis=dict(title="X, м", range=x_scene_range, autorange=False),
            yaxis=dict(title="Y, м", range=y_scene_range, autorange=False),
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


# Сохранение
def save_figure(fig: go.Figure, path: Path) -> None:
    # Сохранить фигуру как standalone HTML (Plotly CDN, без встроенного JS).
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")


# Пакетная отрисовка (процессы)
@dataclass(frozen=True, slots=True)
class WindPlotRenderJob:
    # Задание на построение одного HTML-графика (picklable для ProcessPool).

    dataset_path: Path
    output_dir: Path
    time_ns: int | None  # np.datetime64[ns]; None, первый шаг датасета
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
    # Построить и сохранить график для одного датасета (отдельный процесс).
    from diplom.data.era5_paths import era5_dataset_title, wind_plot_html_path

    name = job.dataset_path.name
    plot_path = wind_plot_html_path(job.dataset_path, job.output_dir)
    logs: list[str] = []

    if plot_path.exists() and not job.force:
        return WindPlotRenderResult(
            dataset_name=name,
            plot_path=plot_path,
            saved=False,
            log_lines=(f"Пропуск {name}: график уже есть -> {plot_path}",),
        )
    if plot_path.exists() and job.force:
        logs.append(f"Пересоздаю {name}: --force -> {plot_path}")

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
    # Выполнить задания последовательно или в пуле процессов.
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
