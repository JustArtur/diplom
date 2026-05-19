"""Быстрая линейная интерполяция на регулярной 4D-сетке (time, pressure, lat, lon)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def _axis_index_weight(axis: np.ndarray, value: float) -> tuple[int, int, float]:
    """Индексы по оси и вес для линейной интерполяции (как scipy linear внутри домена)."""
    n = axis.shape[0]
    if n < 2:
        return 0, 0, 0.0
    if value <= axis[0]:
        return 0, 0, 0.0
    if value >= axis[-1]:
        last = n - 1
        return last - 1, last, 1.0
    i1 = int(np.searchsorted(axis, value, side="right"))
    i0 = i1 - 1
    denom = axis[i1] - axis[i0]
    weight = 0.0 if denom == 0.0 else (value - axis[i0]) / denom
    return i0, i1, weight


def sample_trilinear(
    values: np.ndarray,
    time_axis: np.ndarray,
    pressure_axis: np.ndarray,
    lat_axis: np.ndarray,
    lon_axis: np.ndarray,
    t: float,
    p: float,
    lat: float,
    lon: float,
) -> np.ndarray:
    """Сэмпл 4 каналов (u, v, w_omega, t) в точке регулярной сетки. Shape: (4,)."""
    if _HAS_NUMBA and values.dtype == np.float32:
        out = np.empty(4, dtype=np.float64)
        _sample_trilinear_numba(
            values,
            time_axis,
            pressure_axis,
            lat_axis,
            lon_axis,
            t,
            p,
            lat,
            lon,
            out,
        )
        return out
    return _sample_trilinear_numpy(
        values, time_axis, pressure_axis, lat_axis, lon_axis, t, p, lat, lon
    )


def _sample_trilinear_numpy(
    values: np.ndarray,
    time_axis: np.ndarray,
    pressure_axis: np.ndarray,
    lat_axis: np.ndarray,
    lon_axis: np.ndarray,
    t: float,
    p: float,
    lat: float,
    lon: float,
) -> np.ndarray:
    it0, it1, wt = _axis_index_weight(time_axis, t)
    ip0, ip1, wp = _axis_index_weight(pressure_axis, p)
    ila0, ila1, wla = _axis_index_weight(lat_axis, lat)
    ilo0, ilo1, wlo = _axis_index_weight(lon_axis, lon)

    result = np.zeros(4, dtype=np.float64)
    for dt, wt_side in ((0, 1.0 - wt), (1, wt)):
        t_idx = it0 if dt == 0 else it1
        for dp, wp_side in ((0, 1.0 - wp), (1, wp)):
            p_idx = ip0 if dp == 0 else ip1
            for dla, wla_side in ((0, 1.0 - wla), (1, wla)):
                la_idx = ila0 if dla == 0 else ila1
                for dlo, wlo_side in ((0, 1.0 - wlo), (1, wlo)):
                    lo_idx = ilo0 if dlo == 0 else ilo1
                    weight = wt_side * wp_side * wla_side * wlo_side
                    result += weight * values[t_idx, p_idx, la_idx, lo_idx]
    return result


if _HAS_NUMBA:

    @njit(cache=True)
    def _axis_index_weight_numba(axis, value):
        n = axis.shape[0]
        if n < 2:
            return 0, 0, 0.0
        if value <= axis[0]:
            return 0, 0, 0.0
        if value >= axis[-1]:
            last = n - 1
            return last - 1, last, 1.0
        i1 = np.searchsorted(axis, value)
        i0 = i1 - 1
        denom = axis[i1] - axis[i0]
        weight = 0.0 if denom == 0.0 else (value - axis[i0]) / denom
        return i0, i1, weight

    @njit(cache=True)
    def _sample_trilinear_numba(values, time_axis, pressure_axis, lat_axis, lon_axis, t, p, lat, lon, out):
        it0, it1, wt = _axis_index_weight_numba(time_axis, t)
        ip0, ip1, wp = _axis_index_weight_numba(pressure_axis, p)
        ila0, ila1, wla = _axis_index_weight_numba(lat_axis, lat)
        ilo0, ilo1, wlo = _axis_index_weight_numba(lon_axis, lon)

        for c in range(4):
            acc = 0.0
            for dt in range(2):
                t_idx = it0 if dt == 0 else it1
                wt_side = 1.0 - wt if dt == 0 else wt
                for dp in range(2):
                    p_idx = ip0 if dp == 0 else ip1
                    wp_side = 1.0 - wp if dp == 0 else wp
                    for dla in range(2):
                        la_idx = ila0 if dla == 0 else ila1
                        wla_side = 1.0 - wla if dla == 0 else wla
                        for dlo in range(2):
                            lo_idx = ilo0 if dlo == 0 else ilo1
                            wlo_side = 1.0 - wlo if dlo == 0 else wlo
                            weight = wt_side * wp_side * wla_side * wlo_side
                            acc += weight * values[t_idx, p_idx, la_idx, lo_idx, c]
            out[c] = acc


@dataclass(frozen=True, slots=True)
class RegularGrid4DSampler:
    """Сэмплер по сетке (T, P, Lat, Lon, C)."""

    values: np.ndarray
    time_axis: np.ndarray
    pressure_axis: np.ndarray
    lat_axis: np.ndarray
    lon_axis: np.ndarray

    @classmethod
    def from_channel_first(cls, data: np.ndarray, time_axis: np.ndarray, pressure_axis: np.ndarray, lat_axis: np.ndarray, lon_axis: np.ndarray) -> RegularGrid4DSampler:
        """data: (4, T, P, Lat, Lon) → values (T, P, Lat, Lon, 4)."""
        values = np.moveaxis(np.asarray(data, dtype=np.float32), 0, -1)
        return cls(
            values=values,
            time_axis=np.asarray(time_axis, dtype=np.float64),
            pressure_axis=np.asarray(pressure_axis, dtype=np.float64),
            lat_axis=np.asarray(lat_axis, dtype=np.float64),
            lon_axis=np.asarray(lon_axis, dtype=np.float64),
        )

    def sample(self, t: float, p: float, lat: float, lon: float) -> np.ndarray:
        return sample_trilinear(
            self.values,
            self.time_axis,
            self.pressure_axis,
            self.lat_axis,
            self.lon_axis,
            t,
            p,
            lat,
            lon,
        )
