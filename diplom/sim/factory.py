"""Фабрика сборки симуляции (тонкая обёртка над Simulation)."""

from __future__ import annotations

from diplom.config import SimulationConfig
from diplom.wind.interp import WindInterpolator

from .simulation import Simulation


def create_simulation(
    simulation_config: SimulationConfig,
    wind_interp: WindInterpolator,
    *,
    env_idx: int | None = None,
) -> Simulation:
    """Собрать ``Simulation``: подстановка координат по умолчанию см. ``Simulation.__init__``."""

    return Simulation(simulation_config, wind_interp, env_idx=env_idx)
