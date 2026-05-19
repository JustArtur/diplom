from dataclasses import dataclass
import warnings

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
from diplom.shared_constants import MAX_VERTICAL_SPEED
from diplom.world import WorldBounds
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
    def __init__(
        self,
        config: SimulationConfig,
        wind_interp: WindInterpolator,
        env_idx: int | None = None,
    ) -> None:
        self.wind_interp = wind_interp
        self.world_bounds: WorldBounds = wind_interp.world_bounds
        self.env_idx = env_idx

        self.sim_time = config.balloon.sim_time
        self.position = np.array(config.balloon.initial_position, dtype=np.float32)
        self.target_position = np.array(config.balloon.target_position, dtype=np.float32)
        self.air_weight = np.float32(config.initial_air_weight)

        env_label = f"env_idx={self.env_idx}" if self.env_idx is not None else "env_idx=—"
        print(  # noqa: T201 — явный вывод конфигурации эпизода в CLI/лог
            f"[Simulation] {env_label}: старт (x, y, z) м={self.position.tolist()}, "
            f"цель (x, y, z) м={self.target_position.tolist()}"
        )

        self.vertical_speed = np.float32(0.0)
        self.vertical_acceleration = np.float32(0.0)
        self.energy_spent = np.float32(0.0)
        self.air_density = np.float32(0.0)
        self._warned_world_bounds = False
        self._wind_buf = np.zeros(3, dtype=np.float32)

        wb = self.world_bounds
        self._x_min = float(wb.x_min)
        self._x_max = float(wb.x_max)
        self._y_min = float(wb.y_min)
        self._y_max = float(wb.y_max)
        self._z_min = float(wb.z_min)
        self._z_max = float(wb.z_max)
        self._time_min_ns = int(self.wind_interp.time_min.astype("datetime64[ns]"))
        self._time_max_ns = int(self.wind_interp.time_max.astype("datetime64[ns]"))
        self._time_min = self.wind_interp.time_min
        self._time_max = self.wind_interp.time_max
        self._max_vertical_speed = float(MAX_VERTICAL_SPEED)

    @staticmethod
    def _clamp_scalar(value: float, vmin: float, vmax: float) -> float:
        if value < vmin:
            return vmin
        if value > vmax:
            return vmax
        return value

    def _clamp_time(self) -> None:
        t_ns = int(self.sim_time.astype("datetime64[ns]"))
        if t_ns < self._time_min_ns:
            self.sim_time = self._time_min
        elif t_ns > self._time_max_ns:
            self.sim_time = self._time_max

    def _apply_position(self, proposed_x: float, proposed_y: float, proposed_z: float) -> None:
        self.position[0] = np.float32(self._clamp_scalar(proposed_x, self._x_min, self._x_max))
        self.position[1] = np.float32(self._clamp_scalar(proposed_y, self._y_min, self._y_max))
        self.position[2] = np.float32(self._clamp_scalar(proposed_z, self._z_min, self._z_max))

    def _limit_vertical_speed(self) -> None:
        vs = float(self.vertical_speed)
        limit = self._max_vertical_speed
        if vs < -limit:
            self.vertical_speed = np.float32(-limit)
        elif vs > limit:
            self.vertical_speed = np.float32(limit)

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

        proposed_x = self.position[0] + np.float32(wind.u) * dt
        proposed_y = self.position[1] + np.float32(wind.v) * dt
        proposed_z = self.position[2] + self.vertical_speed * dt
        if not self._warned_world_bounds and (
            proposed_x < self._x_min
            or proposed_x > self._x_max
            or proposed_y < self._y_min
            or proposed_y > self._y_max
            or proposed_z < self._z_min
            or proposed_z > self._z_max
        ):
            self._warned_world_bounds = True
            env_label = f"env_{self.env_idx:03d}" if self.env_idx is not None else "env"
            warnings.warn(
                (
                    f"[{env_label}] Аэростат вышел за границы симуляционного мира; "
                    f"координаты будут клампиться к диапазону "
                    f"[{self._x_min:.0f}, {self._x_max:.0f}] м по X "
                    f"и [{self._y_min:.0f}, {self._y_max:.0f}] м по Y "
                    f"и [{self._z_min:.0f}, {self._z_max:.0f}] м по Z."
                ),
                RuntimeWarning,
                stacklevel=2,
            )

        # Ограничиваем саму высоту (не приращение) снизу и сверху границами вертикали датасета (ISA).
        self._apply_position(float(proposed_x), float(proposed_y), float(proposed_z))
        # Ограничиваем вертикальную скорость, чтобы предотвратить численный взрыв при сильном дисбалансе сил.
        self._limit_vertical_speed()

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
        self._wind_buf[0] = wind.u
        self._wind_buf[1] = wind.v
        self._wind_buf[2] = wind.w
        return SimResult(
            position=self.position,
            target_position=self.target_position,
            vertical_speed=self.vertical_speed,
            vertical_acceleration=self.vertical_acceleration,
            energy_spent=self.energy_spent,
            air_density=self.air_density,
            air_weight=self.air_weight,
            wind=self._wind_buf,
            temperature=wind.temperature,
            pressure=wind.pressure,
        )
