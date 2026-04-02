"""HUD (heads-up display) для визуализации аэростата."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import pyvista as pv


@dataclass(frozen=True)
class HudState:
    """Иммутабельный снимок состояния симуляции для отрисовки HUD."""

    position: np.ndarray                    # позиция аэростата [x, y, z] (м)
    target_position: np.ndarray             # позиция цели [x, y, z] (м)
    # setpoint_altitude: float                # заданная высота (м)
    energy_spent: float                     # затраченная энергия (ед.)
    vertical_speed: float                   # текущая вертикальная скорость (м/с)
    vertical_acceleration: float            # текущее вертикальное ускорение (м/с²)
    last_wind: Tuple[float, float, float]   # последний вектор ветра (u, v, w) м/с
    last_temperature: Optional[float]       # температура воздуха в точке аэростата (K)
    start_monotonic: float                  # отметка monotonic-clock на момент старта

    @property
    def height(self) -> float:
        """Текущая высота аэростата (м)."""
        return float(self.position[2])


class BalloonHUD:
    """Текстовый HUD, отображаемый поверх 3D-сцены PyVista."""

    def __init__(
        self,
        plotter: pv.Plotter,
        *,
        position: Tuple[int, int] = (10, 10),
        font_size: int = 11,
        color: str = "white",
        name: str = "hud",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._plotter = plotter
        self._position = position
        self._font_size = font_size
        self._color = color
        self._name = name
        self._clock = clock

    def update(self, state: HudState) -> None:
        """Перерисовать HUD с актуальным состоянием."""
        self._plotter.add_text(
            self._format(state),
            position=self._position,
            font_size=self._font_size,
            color=self._color,
            name=self._name,
        )

    def _format(self, state: HudState) -> str:
        """Сформировать строку HUD из состояния симуляции."""
        elapsed = self._clock() - state.start_monotonic
        dist = float(np.linalg.norm(state.target_position - state.position))

        wx, wy, wz = state.last_wind
        # |V_h| = √(u² + v²) — модуль горизонтальной скорости ветра
        horiz_speed = float(np.hypot(wx, wy))
        # θ = atan2(u, v) — азимут (курс) ветра, 0° = север, по часовой
        bearing = (float(np.degrees(np.arctan2(wx, wy))) + 360.0) % 360.0
        vert_arrow = "\u2191" if wz >= 0 else "\u2193"
        vert_str = f"{vert_arrow}{abs(wz):.2f}"

        # Температура: K → °C  (T_C = T_K − 273.15)
        if state.last_temperature is not None:
            temp_c = state.last_temperature - 273.15
            temp_str = f"T: {temp_c:.1f} \u00b0C"
        else:
            temp_str = "T: н/д"

        return (
            f"Высота: {state.height:.1f} м |  "
            f"Позиция: ({state.position[0]:.1f}, {state.position[1]:.1f}) м  |  "
            f"Скорость: {state.vertical_speed:.2f} м/с  |  "
            f"Ускорение: {state.vertical_acceleration:.2f} м/с²  |  "
            f"Ветер: {horiz_speed:.1f} м/с, курс: {bearing:5.1f}\u00b0, "
            f"верт: {vert_str} м/с  |  "
            f"{temp_str}  |  "
            f"До цели: {dist:.1f} м  |  "
            f"Затрачено энергии: {state.energy_spent:.0f} ед  |  "
            f"Время: {elapsed:6.1f} c  |  "
            f"I/K \u2014 высота, мышь \u2014 камера"
        )
