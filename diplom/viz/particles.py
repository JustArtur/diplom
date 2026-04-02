"""Частицы-трассеры для визуализации ветрового поля вокруг аэростата."""

from __future__ import annotations

import numpy as np
import pyvista as pv

from diplom.wind.interp import WindInterpolator

from .constants import (
    MIN_HEIGHT,
    NUM_PARTICLES,
    PARTICLE_SPEED,
    STREAK_LENGTH,
    VISIBLE_RADIUS,
    VISIBLE_Z_RANGE,
)


class WindParticles:
    """
    Частицы, визуализирующие ветровое поле штрихами (streak lines).

    Частицы живут в диске радиуса VISIBLE_RADIUS вокруг аэростата
    и респавнятся при выходе за его границы.
    """

    def __init__(self, center: np.ndarray, wind_interpolator: WindInterpolator, time: np.datetime64) -> None:
        self.n_total = NUM_PARTICLES  # количество частиц
        self._wind_interpolator = wind_interpolator
        self._time = time
        self._center = np.array(center, dtype=float)

        # Предвычисленный массив связности линий для VTK PolyData, формат lines [2, head_index, tail_index]
        # head_index, tail_index - индексы из массива pts
        self._lines = np.empty(3 * self.n_total, dtype=np.intp)  # [2, head_index, tail_index]
        self._lines[0::3] = 2
        self._lines[1::3] = np.arange(0, 2 * self.n_total, 2)
        self._lines[2::3] = np.arange(1, 2 * self.n_total, 2)

        # Спавн → сэмпл ветра → построение меша
        self._spawn_all()
        self._sample_wind()
        self.mesh = self._build_mesh()

    # ──────────────────── Вспомогательные ────────────────────

    def _z_bounds(self, center_z: float) -> tuple[float, float]:
        """Допустимый диапазон высот [lo, hi] для спавна и отсечения."""
        return max(center_z - VISIBLE_Z_RANGE, MIN_HEIGHT), center_z + VISIBLE_Z_RANGE

    def _random_disk(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Равномерное распределение *n* точек на диске радиуса VISIBLE_RADIUS.

        r = √u · R — корень компенсирует рост площади круга (~ r²),
        чтобы плотность по площади была равномерной.
        """
        angles = np.random.uniform(0, 2 * np.pi, n)  # Определяем угол
        # Определяем радиус (корень для равномерного распределения, иначе будет больше плотности в центре)
        radii = np.sqrt(np.random.uniform(0, 1, n)) * VISIBLE_RADIUS
        return radii * np.cos(angles), radii * np.sin(angles)  # Переводим в x и y

    # ──────────────────── Ветер ────────────────────

    def _sample_wind(self) -> None:
        """Сэмплировать ветер в текущих позициях частиц через WindInterpolator."""
        vec = self._wind_interpolator.batch_vector_at(
            self.position[:, 0],
            self.position[:, 1],
            self.position[:, 2],
            time=np.array([self._time] * len(self.position)),
        )
        self._last_wind = vec[:, 0], vec[:, 1], vec[:, 2]

    # ──────────────────── Спавн ────────────────────

    def _spawn_all(self) -> None:
        """Создать начальные позиции всех частиц вокруг центра."""
        cx, cy, cz = self._center
        z_lo, z_hi = self._z_bounds(cz)

        xs, ys = self._random_disk(self.n_total)
        xs += cx
        ys += cy
        zs = np.random.uniform(z_lo, z_hi, self.n_total)
        self.position = np.column_stack([xs, ys, zs])

    # ──────────────────── Шаг симуляции ────────────────────

    def step(self, center: np.ndarray, time: np.datetime64) -> None:
        """Продвинуть частицы по ветру, респавнить вышедшие, обновить меш."""
        self._time = time
        center = np.array(center, dtype=float)
        shift = center - self._center
        if np.any(shift):
            # Переносим всё "облако" частиц за аэростатом
            self.position += shift
            self._center = center
        cx, cy, cz = self._center

        # Сэмплируем 3D-ветер и двигаем частицы
        self._sample_wind()
        wx, wy, wz = self._last_wind
        self.position[:, 0] += wx * PARTICLE_SPEED
        self.position[:, 1] += wy * PARTICLE_SPEED
        self.position[:, 2] += wz * PARTICLE_SPEED

        # Определяем частицы, вылетевшие за пределы видимости
        z_lo, z_hi = self._z_bounds(cz)
        dist = np.sqrt((self.position[:, 0] - cx) ** 2 + (self.position[:, 1] - cy) ** 2)
        out = (dist > VISIBLE_RADIUS) | (self.position[:, 2] < z_lo) | (self.position[:, 2] > z_hi)

        # Респавн вышедших частиц вокруг текущей позиции аэростата
        n_out = int(out.sum())
        if n_out > 0:
            xs, ys = self._random_disk(n_out)
            self.position[out, 0] = xs + cx
            self.position[out, 1] = ys + cy
            self.position[out, 2] = np.random.uniform(z_lo, z_hi, n_out)

        self._update_mesh()

    # ──────────────────── Меш (приватные) ────────────────────

    def _build_mesh(self) -> pv.PolyData:
        """Создать новый PolyData со штрихами (вызывается один раз)."""
        pts, speed = self._streak_data()
        mesh = pv.PolyData(pts, lines=self._lines)
        mesh["speed"] = speed
        return mesh

    def _update_mesh(self) -> None:
        """Обновить существующий меш in-place (без пересоздания VTK-актора)."""
        pts, speed = self._streak_data()
        self.mesh.points = pts
        self.mesh["speed"] = speed
        self.mesh.Modified()

    def _streak_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Вычислить пары точек [голова, хвост] и скаляры скорости ветра."""
        wx, wy, wz = self._last_wind
        pts = np.empty((2 * self.n_total, 3))
        # Чётные индексы — голова (текущая позиция)

        pts[0::2] = self.position
        # Нечётные — хвост (позиция − вектор ветра × длину штриха)
        pts[1::2, 0] = self.position[:, 0] - wx * STREAK_LENGTH
        pts[1::2, 1] = self.position[:, 1] - wy * STREAK_LENGTH
        pts[1::2, 2] = self.position[:, 2] - wz * STREAK_LENGTH
        # |V| = √(u² + v² + w²)
        speed_per_particle = np.sqrt(wx ** 2 + wy ** 2 + wz ** 2)
        speed = np.repeat(speed_per_particle, 2)
        return pts, speed
