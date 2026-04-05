from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from diplom.sim.constants import (
    AIR_DRAG_COEFFICIENT,
    AIR_MOLAR_MASS,
    BALLOON_CROSS_SECTION,
    BALLOON_VOLUME,
    BALLOON_WEIGHT,
    DEFAULT_AIR_WEIGHT,
    GAS_CONSTANT,
    GRAVITY_ACCELERATION, INITIAL_POSITION, TARGET_POSITION, SIM_TIME,
)
from diplom.wind.interp import WindInterpolator as WindInterp


@dataclass
class SimParams:
    wind_interp: WindInterp
    sim_time: datetime = SIM_TIME
    initial_position: np.ndarray = field(default_factory=lambda: INITIAL_POSITION.copy())
    target_position: np.ndarray = field(default_factory=lambda: TARGET_POSITION.copy())
    initial_air_weight: float = DEFAULT_AIR_WEIGHT


@dataclass
class SimResult:
    position: np.ndarray
    target_position: np.ndarray
    vertical_speed: float
    vertical_acceleration: float
    energy_spent: float
    air_density: float
    air_weight: float
    wind: np.ndarray
    temperature: float
    pressure: float


class Simulation:
    def __init__(self, ps: SimParams):
        self.wind_interp = ps.wind_interp  # Интерполяция ветра

        self.sim_time = ps.sim_time  # Время симуляции
        self.position = ps.initial_position  # Текущая позиция
        self.target_position = ps.target_position  # Позиция цели
        self.air_weight = ps.initial_air_weight  # Кол во закаченного воздуха в старостат

        self.vertical_speed = 0.0  # Вертикальная скорость
        self.vertical_acceleration = 0.0  # Вертикальное ускорение
        self.energy_spent = 0.0  # Потраченная энергия(на закачку воздуха)
        self.air_density = 0.0
        self.pressure = 0.0
        self.temperature = 0.0

    def step(self, dt: float, air_pump_speed: float) -> SimResult:
        """Снос аэростата ветром (горизонтальный + вертикальный).

        Интегрирование методом Эйлера (первого порядка):
            x(t+dt) = x(t) + v_x · k · dt
            y(t+dt) = y(t) + v_y · k · dt
            z(t+dt) = z(t) + v_z · k · dt
        """
        # np.datetime64 не умеет складываться с float, поэтому переводим dt в timedelta.
        self.sim_time += np.timedelta64(int(dt), "s")
        wx, wy, wz, temperature, pressure = self.wind_interp.vector_at(self.position[0], self.position[1],
                                                                       self.position[2], self.sim_time)

        air_mass_delta = air_pump_speed * dt

        self.air_weight = max(0.0, self.air_weight + air_mass_delta)
        self.compute_vertical_speed(pressure, temperature, dt)
        self.energy_spent += abs(air_mass_delta) * 10.0

        self.position[0] += wx * dt
        self.position[1] += wy * dt
        self.position[2] += (wz * dt) + self.vertical_speed * dt

        return SimResult(
            position=np.array(self.position, dtype=float),
            target_position=np.array(self.target_position, dtype=float),
            wind=np.array([wx, wy, wz], dtype=float),
            vertical_speed=self.vertical_speed,
            vertical_acceleration=self.vertical_acceleration,
            energy_spent=self.energy_spent,
            air_density=self.air_density,
            air_weight=self.air_weight,
            temperature=self.temperature,
            pressure=self.pressure
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
        self.air_density = self.gas_density(AIR_MOLAR_MASS, pressure, temperature)
        total_mass = BALLOON_WEIGHT + self.air_weight
        archimedes_force = self.air_density * GRAVITY_ACCELERATION * BALLOON_VOLUME
        weight_force = total_mass * GRAVITY_ACCELERATION
        drag_force = self._drag_force(self.air_density)
        self.vertical_acceleration = (archimedes_force - weight_force - drag_force) / total_mass

        self.vertical_speed += self.vertical_acceleration * dt

    def _drag_force(self, air_density: float) -> float:
        """Лобовое сопротивление по вертикали, направленное против движения."""
        speed = float(self.vertical_speed)
        if abs(speed) < 1e-9:
            return 0.0
        drag = 0.5 * air_density * AIR_DRAG_COEFFICIENT * BALLOON_CROSS_SECTION * speed ** 2
        return drag if speed > 0 else -drag
