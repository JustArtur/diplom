from dataclasses import dataclass

import numpy as np

from diplom.sim.constants import (
    AIR_DRAG_COEFFICIENT,
    AIR_MOLAR_MASS,
    BALLOON_CROSS_SECTION,
    BALLOON_VOLUME,
    BALLOON_WEIGHT,
    DEFAULT_AIR_WEIGHT,
    DRIFT_SPEED_SCALE,
    GAS_CONSTANT,
    GRAVITY_ACCELERATION,
)
from diplom.wind.interp import WindInterpolator as WindInterp


@dataclass
class SimParams:
    position: np.ndarray
    dt: float
    air_pump_speed: float
    sim_time: np.datetime64
    energy: float


@dataclass
class SimState:
    position: np.ndarray
    vertical_speed: float
    vertical_acceleration: float
    energy_spent: float


class Simulation:
    def __init__(self, wind_interp: WindInterp):
        self.air_weight = DEFAULT_AIR_WEIGHT  # Кол во закаченного воздуха в старостат
        self.vertical_speed = 0  # Вертикальная скорость
        self.vertical_acceleration = 0.0  # Вертикальное ускорение
        self.energy_spent = 0  # Потраченная энергия(на закачку воздуха)

        self.wind_interp = wind_interp  # Интерполяция ветра
        # self.position = ps.position  # Текущая позиция
        # self.dt = ps.dt  # Разница времени для расчета данных
        # self.air_pump_speed = ps.air_pump_speed  # Скорость закачки воздуха в баллон
        # self.sim_time = ps.sim_time  # Время симуляции

    def step(self, ps: SimParams):
        """Снос аэростата ветром (горизонтальный + вертикальный).

        Интегрирование методом Эйлера (первого порядка):
            x(t+dt) = x(t) + v_x · k · dt
            y(t+dt) = y(t) + v_y · k · dt
            z(t+dt) = z(t) + v_z · k · dt
        где k = DRIFT_SPEED_SCALE — масштабный коэффициент визуализации.
        """
        wx, wy, wz, temperature, pressure = self.wind_interp.vector_at(ps.position[0], ps.position[1],
                                                                       ps.position[2], ps.sim_time)

        air_mass_delta = ps.air_pump_speed * ps.dt
        self.air_weight = max(0.0, self.air_weight + air_mass_delta)
        self.compute_vertical_speed(pressure, temperature, ps.dt)
        self.energy_spent += abs(air_mass_delta) * 10.0
        # x(t+dt) = x(t) + v · k · dt
        ps.position[0] += wx * DRIFT_SPEED_SCALE * ps.dt
        ps.position[1] += wy * DRIFT_SPEED_SCALE * ps.dt
        ps.position[2] += (wz * DRIFT_SPEED_SCALE * ps.dt) + self.vertical_speed * ps.dt

        return SimState(
            position=np.array(ps.position, dtype=float),
            vertical_speed=self.vertical_speed,
            vertical_acceleration=self.vertical_acceleration,
            energy_spent=self.energy_spent,
        )

    def gas_density(self, molar_mass, pressure, temperature):
        """
        Из уравнения Менделеева-Клапейрона выводим формулу плотности газа:
        AIR_DENSITY = AIR_PRESSURE*AIR_MOLAR_MASS/(GAS_CONSTANT*AIR_TEMPERATURE(Kelvin))
        """
        pressure_pa = pressure * 100.0
        return (pressure_pa * molar_mass) / (GAS_CONSTANT * temperature)

    def compute_vertical_speed(self, pressure, temperature, dt):
        """
        По второму закону Ньютона:
        a = (F_archimedes - F_weight - F_drag) / m
        """
        air_density = self.gas_density(AIR_MOLAR_MASS, pressure, temperature)
        total_mass = BALLOON_WEIGHT + self.air_weight
        archimedes_force = air_density * GRAVITY_ACCELERATION * BALLOON_VOLUME
        weight_force = total_mass * GRAVITY_ACCELERATION
        drag_force = self._drag_force(air_density)
        self.vertical_acceleration = (archimedes_force - weight_force - drag_force) / total_mass

        self.vertical_speed += self.vertical_acceleration * dt

    def _drag_force(self, air_density: float) -> float:
        """Лобовое сопротивление по вертикали, направленное против движения."""
        speed = float(self.vertical_speed)
        if abs(speed) < 1e-9:
            return 0.0
        drag = 0.5 * air_density * AIR_DRAG_COEFFICIENT * BALLOON_CROSS_SECTION * speed ** 2
        return drag if speed > 0 else -drag
