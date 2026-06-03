# Локальный рельеф и декорации (кусты) для PyVista-визуализации.

from __future__ import annotations

import numpy as np
import pyvista as pv

from .constants import (
    BUSH_COUNT,
    BUSH_EXCLUDE_RADIUS_M,
    BUSH_FOLIAGE_RGB,
    BUSH_PLACE_RADIUS_M,
    BUSH_SCALE_MAX,
    BUSH_SCALE_MIN,
    BUSH_SEED,
    BUSH_TRUNK_RGB,
    TERRAIN_AMP_COS,
    TERRAIN_AMP_SIN,
    TERRAIN_FREQ_COS,
    TERRAIN_FREQ_SIN,
    TERRAIN_GRASS_RGB,
    TERRAIN_PATCH_SIZE_M,
    TERRAIN_RESOLUTION,
)


def terrain_height_at(x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray | float:
    # Высота синтетического рельефа в метрах (та же формула, что у плоскости земли).
    return (
        TERRAIN_AMP_SIN * np.sin(np.asarray(y) * TERRAIN_FREQ_SIN)
        + TERRAIN_AMP_COS * np.cos(np.asarray(x) * TERRAIN_FREQ_COS)
    )


def build_terrain_plane() -> pv.PolyData:
    # Зелёная плоскость с лёгким рельефом в локальных координатах сцены.
    plane = pv.Plane(
        center=(0.0, 0.0, 0.0),
        i_size=TERRAIN_PATCH_SIZE_M,
        j_size=TERRAIN_PATCH_SIZE_M,
        i_resolution=TERRAIN_RESOLUTION,
        j_resolution=TERRAIN_RESOLUTION,
    )
    x_pts, y_pts = plane.points[:, 0], plane.points[:, 1]
    plane.points[:, 2] = terrain_height_at(x_pts, y_pts)
    plane.point_data["colors"] = np.tile(
        np.asarray(TERRAIN_GRASS_RGB, dtype=np.uint8),
        (plane.n_points, 1),
    )
    return plane


def _mesh_with_vertex_colors(mesh: pv.DataSet, rgb: tuple[int, int, int]) -> pv.DataSet:
    colored = mesh.copy(deep=True)
    colored.point_data["colors"] = np.tile(np.asarray(rgb, dtype=np.uint8), (colored.n_points, 1))
    return colored


def _single_bush(
    x: float,
    y: float,
    rng: np.random.Generator,
) -> list[pv.DataSet]:
    # Один куст: цилиндр-ствол и несколько конусов листвы.
    z = float(terrain_height_at(x, y))
    scale = float(rng.uniform(BUSH_SCALE_MIN, BUSH_SCALE_MAX))
    height = scale * float(rng.uniform(5.0, 9.0))
    radius = scale * float(rng.uniform(2.5, 4.5))
    trunk_h = height * 0.3

    trunk = pv.Cylinder(
        center=(x, y, z + trunk_h * 0.5),
        direction=(0.0, 0.0, 1.0),
        radius=scale * 0.45,
        height=trunk_h,
        resolution=8,
    )
    parts: list[pv.DataSet] = [_mesh_with_vertex_colors(trunk, BUSH_TRUNK_RGB)]

    foliage_count = int(rng.integers(2, 5))
    for _ in range(foliage_count):
        dx = float(rng.uniform(-0.55, 0.55) * radius)
        dy = float(rng.uniform(-0.55, 0.55) * radius)
        cone_h = height * float(rng.uniform(0.45, 0.7))
        cone_r = radius * float(rng.uniform(0.75, 1.1))
        base_z = z + trunk_h + cone_h * 0.15
        cone = pv.Cone(
            center=(x + dx, y + dy, base_z + cone_h * 0.5),
            direction=(0.0, 0.0, 1.0),
            height=cone_h,
            radius=cone_r,
            resolution=10,
        )
        parts.append(_mesh_with_vertex_colors(cone, BUSH_FOLIAGE_RGB))

    return parts


def build_bushes_mesh(seed: int = BUSH_SEED) -> pv.PolyData:
    # Случайные кусты в кольце вокруг аэростата (хорошо видны с камеры).
    rng = np.random.default_rng(seed)
    exclude_r = BUSH_EXCLUDE_RADIUS_M
    place_r = BUSH_PLACE_RADIUS_M

    parts: list[pv.DataSet] = []
    placed = 0
    attempts = 0
    max_attempts = BUSH_COUNT * 40

    while placed < BUSH_COUNT and attempts < max_attempts:
        attempts += 1
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        radius = float(rng.uniform(exclude_r, place_r))
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        parts.extend(_single_bush(x, y, rng))
        placed += 1

    if not parts:
        return pv.PolyData()

    return pv.merge(parts)
