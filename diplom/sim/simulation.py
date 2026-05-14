from dataclasses import dataclass

import numpy as np

from diplom.config import SimulationConfig
from diplom.sim.constants import (
    AIR_DRAG_COEFFICIENT,
    AIR_MOLAR_MASS,
    BALLOON_CROSS_SECTION,
    BALLOON_VOLUME,
    BALLOON_WEIGHT,
    ENERGY_COST_PER_KG,
    GAS_CONSTANT,
    GRAVITY_ACCELERATION,
)
from diplom.shared_constants import MAX_HEIGHT, MAX_VERTICAL_SPEED, MIN_HEIGHT, WORLD_SIZE
from diplom.wind.interp import WindInterpolator, WindSample


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
    def __init__(self, config: SimulationConfig, wind_interp: WindInterpolator) -> None:
        self.wind_interp = wind_interp

        self.sim_time = config.balloon.sim_time
        self.position = np.array(config.balloon.initial_position, dtype=np.float32)
        self.target_position = np.array(config.balloon.target_position, dtype=np.float32)
        self.air_weight = np.float32(config.initial_air_weight)

        self.vertical_speed = np.float32(0.0)
        self.vertical_acceleration = np.float32(0.0)
        self.energy_spent = np.float32(0.0)
        self.air_density = np.float32(0.0)

    def _clamp_time(self) -> None:
        self.sim_time = np.clip(self.sim_time, self.wind_interp.time_min, self.wind_interp.time_max)

    def snapshot(self) -> SimResult:
        """Собрать текущее состояние без шага по времени."""
        self._clamp_time()
        wind = self.interpolate_wind()
        return self._build_result(wind)

    def step(self, dt: float, air_pump_speed: float) -> SimResult:
        """Снос аэростата ветром (горизонтальный + вертикальный).

        Интегрирование методом Эйлера (первого порядка):
            x(t+dt) = x(t) + v_x · k · dt
            y(t+dt) = y(t) + v_y · k · dt
            z(t+dt) = z(t) + v_z · k · dt
        """
        # np.datetime64 не умеет складываться с float, поэтому переводим dt в timedelta.
        self.sim_time += np.timedelta64(int(dt), "s")
        self._clamp_time()
        wind = self.interpolate_wind()

        air_mass_delta = np.float32(air_pump_speed * dt)

        self.air_weight = np.maximum(np.float32(0.0), self.air_weight + air_mass_delta)
        self._compute_vertical_speed(wind.pressure, wind.temperature, wind.w, dt)
        self.energy_spent += np.abs(air_mass_delta) * np.float32(ENERGY_COST_PER_KG)

        self.position[0] = np.clip(self.position[0] + np.float32(wind.u) * dt, 0.0, WORLD_SIZE)
        self.position[1] = np.clip(self.position[1] + np.float32(wind.v) * dt, 0.0, WORLD_SIZE)
        # Ограничиваем саму высоту (не приращение) снизу и сверху допустимым диапазоном ERA5.
        self.position[2] = np.clip(self.position[2] + self.vertical_speed * dt, MIN_HEIGHT, MAX_HEIGHT)
        # Ограничиваем вертикальную скорость, чтобы предотвратить численный взрыв при сильном дисбалансе сил.
        self.vertical_speed = np.clip(self.vertical_speed, -MAX_VERTICAL_SPEED, MAX_VERTICAL_SPEED)

        return self._build_result(wind)

    def gas_density(self, molar_mass: float, pressure: float, temperature: float) -> float:
        """
        Из уравнения Менделеева-Клапейрона выводим формулу плотности газа:
        AIR_DENSITY = AIR_PRESSURE*AIR_MOLAR_MASS/(GAS_CONSTANT*AIR_TEMPERATURE(Kelvin))
        """
        pressure_pa = np.float32(pressure) * np.float32(100.0)
        return np.float32((pressure_pa * molar_mass) / (GAS_CONSTANT * temperature))

    def _compute_vertical_speed(self, pressure: float, temperature: float, wz: float, dt: float) -> None:
        """
        По второму закону Ньютона:
        a = (F_archimedes - F_weight - F_drag) / m
        """
        # Физика баллона считается по плотности воздуха и силам Архимеда/тяжести/сопротивления.
        self.air_density = np.float32(self.gas_density(AIR_MOLAR_MASS, pressure, temperature))
        total_mass = np.float32(BALLOON_WEIGHT) + self.air_weight
        archimedes_force = self.air_density * np.float32(GRAVITY_ACCELERATION) * np.float32(BALLOON_VOLUME)
        weight_force = total_mass * np.float32(GRAVITY_ACCELERATION)
        drag_force = np.float32(self._drag_force(float(self.air_density)))
        self.vertical_acceleration = np.float32((archimedes_force - weight_force - drag_force) / total_mass)

        self.vertical_speed = np.float32(self.vertical_speed + self.vertical_acceleration * dt + np.float32(wz))

    def _drag_force(self, air_density: float) -> float:
        """Лобовое сопротивление по вертикали, направленное против движения."""
        speed = float(self.vertical_speed)
        if abs(speed) < 1e-9:
            return 0.0
        drag = 0.5 * air_density * AIR_DRAG_COEFFICIENT * BALLOON_CROSS_SECTION * speed ** 2
        return drag if speed > 0 else -drag

    def interpolate_wind(self) -> WindSample:
        return self.wind_interp.vector_at(self.position[0], self.position[1], self.position[2], self.sim_time)

    def _build_result(self, wind: WindSample) -> SimResult:
        """Собрать объект результата из текущего внутреннего состояния."""
        return SimResult(
            position=np.array(self.position, dtype=np.float32),
            target_position=np.array(self.target_position, dtype=np.float32),
            vertical_speed=self.vertical_speed,
            vertical_acceleration=self.vertical_acceleration,
            energy_spent=self.energy_spent,
            air_density=self.air_density,
            air_weight=self.air_weight,
            wind=np.array([wind.u, wind.v, wind.w], dtype=np.float32),
            temperature=wind.temperature,
            pressure=wind.pressure,
        )
