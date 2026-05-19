"""Сравнение scipy RegularGridInterpolator и быстрого trilinear на случайных точках."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from diplom.config import WindConfig
from diplom.wind.interp import WindInterpolator, WindSample, omega_to_w_mps_scalar
from diplom.wind.trilinear import RegularGrid4DSampler, _HAS_NUMBA


@dataclass(frozen=True, slots=True)
class ComparisonRow:
    run_id: int
    seed: int
    n_points: int
    max_abs_u: float
    max_abs_v: float
    max_abs_w: float
    max_abs_temp: float
    max_abs_pressure: float
    mean_abs_u: float
    mean_abs_v: float
    mean_abs_w: float
    mean_abs_temp: float
    rmse_u: float
    rmse_v: float
    rmse_w: float
    rmse_temp: float


def _vector_at_scipy(interp: WindInterpolator, x: float, y: float, z: float, time: np.datetime64) -> WindSample:
  lat, lon = interp._xy_to_latlon_scalar(x, y)
  level = interp._z_to_pressure_scalar(z)
  t = interp._time_to_float_scalar(time)
  pt = interp._pt_buf
  pt[0, 0] = t
  pt[0, 1] = level
  pt[0, 2] = lat
  pt[0, 3] = lon
  u, v, w_omega, temp = interp._interp(pt)[0]
  w = omega_to_w_mps_scalar(float(w_omega), level, float(temp))
  return WindSample(u=float(u), v=float(v), w=w, temperature=float(temp), pressure=level)


def _vector_at_trilinear(
    interp: WindInterpolator,
    sampler: RegularGrid4DSampler,
    x: float,
    y: float,
    z: float,
    time: np.datetime64,
) -> WindSample:
    lat, lon = interp._xy_to_latlon_scalar(x, y)
    level = interp._z_to_pressure_scalar(z)
    t = interp._time_to_float_scalar(time)
    u, v, w_omega, temp = sampler.sample(t, level, lat, lon)
    w = omega_to_w_mps_scalar(float(w_omega), level, float(temp))
    return WindSample(u=float(u), v=float(v), w=w, temperature=float(temp), pressure=level)


def _random_points(
    rng: np.random.Generator,
    interp: WindInterpolator,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    wb = interp.world_bounds
    x = rng.uniform(wb.x_min, wb.x_max, size=n)
    y = rng.uniform(wb.y_min, wb.y_max, size=n)
    z = rng.uniform(wb.z_min, wb.z_max, size=n)
    t_min_ns = int(interp.time_min.astype("datetime64[ns]"))
    t_max_ns = int(interp.time_max.astype("datetime64[ns]"))
    t_ns = rng.integers(t_min_ns, t_max_ns + 1, size=n, endpoint=True)
    times = t_ns.astype("datetime64[ns]")
    return x, y, z, times


def _compare_batch(
    interp: WindInterpolator,
    sampler: RegularGrid4DSampler,
    *,
    run_id: int,
    seed: int,
    n_points: int,
) -> ComparisonRow:
    rng = np.random.default_rng(seed)
    x, y, z, times = _random_points(rng, interp, n_points)

    du = np.empty(n_points, dtype=np.float64)
    dv = np.empty(n_points, dtype=np.float64)
    dw = np.empty(n_points, dtype=np.float64)
    dt = np.empty(n_points, dtype=np.float64)
    dp = np.empty(n_points, dtype=np.float64)

    for i in range(n_points):
        old = _vector_at_scipy(interp, float(x[i]), float(y[i]), float(z[i]), times[i])
        new = _vector_at_trilinear(interp, sampler, float(x[i]), float(y[i]), float(z[i]), times[i])
        du[i] = abs(new.u - old.u)
        dv[i] = abs(new.v - old.v)
        dw[i] = abs(new.w - old.w)
        dt[i] = abs(new.temperature - old.temperature)
        dp[i] = abs(new.pressure - old.pressure)

    return ComparisonRow(
        run_id=run_id,
        seed=seed,
        n_points=n_points,
        max_abs_u=float(du.max()),
        max_abs_v=float(dv.max()),
        max_abs_w=float(dw.max()),
        max_abs_temp=float(dt.max()),
        max_abs_pressure=float(dp.max()),
        mean_abs_u=float(du.mean()),
        mean_abs_v=float(dv.mean()),
        mean_abs_w=float(dw.mean()),
        mean_abs_temp=float(dt.mean()),
        rmse_u=float(np.sqrt(np.mean(du**2))),
        rmse_v=float(np.sqrt(np.mean(dv**2))),
        rmse_w=float(np.sqrt(np.mean(dw**2))),
        rmse_temp=float(np.sqrt(np.mean(dt**2))),
    )


def _make_synthetic_interpolator() -> WindInterpolator:
    """Синтетическая сетка ERA5-подобной формы, если NetCDF недоступен."""
    n_t, n_p, n_lat, n_lon = 8, 12, 16, 20
    time_axis_ns = np.linspace(0, 7 * 24 * 3600, n_t, dtype=np.int64) * 1_000_000_000
    pressure_axis_hpa = np.linspace(1000.0, 200.0, n_p)
    lat_axis = np.linspace(48.0, 52.0, n_lat)
    lon_axis = np.linspace(8.0, 12.0, n_lon)

    tt, pp, la, lo = np.meshgrid(
        time_axis_ns.astype(np.float64) * 1e-9,
        pressure_axis_hpa,
        lat_axis,
        lon_axis,
        indexing="ij",
    )
    u = 10.0 * np.sin(la * 0.3) * np.cos(lo * 0.2) + 0.01 * pp
    v = 8.0 * np.cos(la * 0.25) * np.sin(lo * 0.15) - 0.005 * pp
    w = 0.5 * np.sin(tt * 0.1) * np.cos(pp * 0.01)
    temp = 288.15 + 0.02 * pp + 2.0 * np.sin(la)

    data = np.stack([u, v, w, temp], axis=0).astype(np.float32)
    return WindInterpolator(
        data=data,
        env_idx=None,
        origin_lat=float(lat_axis[0]),
        origin_lon=float(lon_axis[0]),
        time_axis_ns=time_axis_ns,
        pressure_axis_hpa=pressure_axis_hpa,
        latitude_axis_deg=lat_axis,
        longitude_axis_deg=lon_axis,
    )


def _load_interpolator(wind_path: Path | None) -> tuple[WindInterpolator, str]:
    path = wind_path or WindConfig().path
    if path.is_file():
        return WindInterpolator.from_file(path), f"ERA5 cache from {path}"
    return _make_synthetic_interpolator(), "synthetic grid (NetCDF not found)"


def run_comparison(
    *,
    wind_path: Path | None = None,
    n_runs: int = 12,
    n_points_per_run: int = 2_000,
    base_seed: int = 42,
) -> list[ComparisonRow]:
    interp, source = _load_interpolator(wind_path)
    sampler = RegularGrid4DSampler.from_channel_first(
        interp.data,
        interp._time_axis_float,
        interp.pressure_axis_hpa,
        interp.latitude_axis_deg,
        interp.longitude_axis_deg,
    )

    print(f"Data source: {source}")
    print(f"Grid shape (T, P, Lat, Lon): {sampler.values.shape[:4]}")
    print(f"Numba enabled: {_HAS_NUMBA}")
    print(f"World bounds: x=[{interp.world_bounds.x_min:.0f}, {interp.world_bounds.x_max:.0f}] "
          f"y=[{interp.world_bounds.y_min:.0f}, {interp.world_bounds.y_max:.0f}] "
          f"z=[{interp.world_bounds.z_min:.0f}, {interp.world_bounds.z_max:.0f}]")
    print()

    header = (
        "run seed     n     "
        "max_u      max_v      max_w      max_T      max_p      "
        "mean_u     mean_v     mean_w     mean_T     "
        "rmse_u     rmse_v     rmse_w     rmse_T"
    )
    print(header)
    print("-" * len(header))

    rows: list[ComparisonRow] = []
    for run_id in range(1, n_runs + 1):
        seed = base_seed + run_id * 997
        row = _compare_batch(interp, sampler, run_id=run_id, seed=seed, n_points=n_points_per_run)
        rows.append(row)
        print(
            f"{row.run_id:3d} {row.seed:5d} {row.n_points:5d} "
            f"{row.max_abs_u:10.3e} {row.max_abs_v:10.3e} {row.max_abs_w:10.3e} "
            f"{row.max_abs_temp:10.3e} {row.max_abs_pressure:10.3e} "
            f"{row.mean_abs_u:10.3e} {row.mean_abs_v:10.3e} {row.mean_abs_w:10.3e} {row.mean_abs_temp:10.3e} "
            f"{row.rmse_u:10.3e} {row.rmse_v:10.3e} {row.rmse_w:10.3e} {row.rmse_temp:10.3e}"
        )

    print()
    print("--- aggregate over all runs ---")
    max_u = max(r.max_abs_u for r in rows)
    max_v = max(r.max_abs_v for r in rows)
    max_w = max(r.max_abs_w for r in rows)
    max_t = max(r.max_abs_temp for r in rows)
    mean_rmse_w = float(np.mean([r.rmse_w for r in rows]))
    print(f"worst max_abs: u={max_u:.6e} v={max_v:.6e} w={max_w:.6e} T={max_t:.6e}")
    print(f"mean rmse_w across runs: {mean_rmse_w:.6e}")

    # warmup numba
    if _HAS_NUMBA:
        x0, y0, z0, t0 = _random_points(np.random.default_rng(0), interp, 3)
        for i in range(3):
            _vector_at_trilinear(interp, sampler, float(x0[i]), float(y0[i]), float(z0[i]), t0[i])

    n_bench = 5_000
    x, y, z, times = _random_points(np.random.default_rng(123), interp, n_bench)

    t0 = time.perf_counter()
    for i in range(n_bench):
        _vector_at_scipy(interp, float(x[i]), float(y[i]), float(z[i]), times[i])
    scipy_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for i in range(n_bench):
        _vector_at_trilinear(interp, sampler, float(x[i]), float(y[i]), float(z[i]), times[i])
    fast_s = time.perf_counter() - t0

    print()
    print(f"--- timing ({n_bench} points, full vector_at path) ---")
    print(f"scipy:     {scipy_s:.3f} s  ({scipy_s / n_bench * 1e6:.1f} µs/call)")
    print(f"trilinear: {fast_s:.3f} s  ({fast_s / n_bench * 1e6:.1f} µs/call)")
    print(f"speedup:   {scipy_s / fast_s:.2f}x")

    interp.close()
    return rows


def main() -> None:
    run_comparison()


if __name__ == "__main__":
    main()
