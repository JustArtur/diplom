"""Общие геодезические и барометрические приближения (lat/lon → м, высота ↔ давление ISA)."""

from __future__ import annotations

import math

import numpy as np


def altitude_to_pressure_hpa(height_m: np.ndarray) -> np.ndarray:
    """Давление (гПа) по высоте через барометрическую формулу стандартной атмосферы (ISA).

    Ниже ~44.3 км: p(h) = p₀ · (1 − L·h / T₀) ^ (g·M / (R·L)).
    """
    p0 = np.float32(1013.25)
    T0 = np.float32(288.15)
    g = np.float32(9.80665)
    L = np.float32(0.0065)
    R = np.float32(8.31447)
    M = np.float32(0.0289644)

    height = np.asarray(height_m, dtype=np.float32)
    max_height = (T0 / L) - np.float32(1e-6)
    height = np.clip(height, 0.0, max_height)
    exponent = (g * M) / (R * L)
    return np.asarray(p0 * np.power(np.float32(1.0) - (L * height) / T0, exponent), dtype=np.float32)


def pressure_hpa_to_altitude_m(pressure_hpa: np.ndarray) -> np.ndarray:
    """Высота (м) по давлению (гПа), обратная к `altitude_to_pressure_hpa` в диапазоне ISA."""
    p0 = np.float32(1013.25)
    T0 = np.float32(288.15)
    g = np.float32(9.80665)
    L = np.float32(0.0065)
    R = np.float32(8.31447)
    M = np.float32(0.0289644)
    exponent = (g * M) / (R * L)

    p = np.asarray(pressure_hpa, dtype=np.float32)
    p = np.clip(p, np.float32(1e-6), p0)
    ratio = p / p0
    h = (T0 / L) * (np.float32(1.0) - np.power(ratio, np.float32(1.0) / exponent))
    max_height = (T0 / L) - np.float32(1e-6)
    return np.clip(h, np.float32(0.0), max_height).astype(np.float32)


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
