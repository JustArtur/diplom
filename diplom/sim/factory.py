"""Фабрика сборки симуляции (тонкая обёртка над Simulation)."""

from __future__ import annotations

from dataclasses import replace

from diplom.config import SimulationConfig
from diplom.world import resolve_balloon_config
from diplom.wind.interp import WindInterpolator

from .simulation import Simulation


def create_simulation(
    simulation_config: SimulationConfig,
    wind_interp: WindInterpolator,
    *,
    env_idx: int | None = None,
) -> Simulation:
    """Собрать ``Simulation``; координаты из ``BalloonConfig`` подставляются из границ датасета, если они не заданы."""

    resolved = replace(
        simulation_config,
        balloon=resolve_balloon_config(simulation_config.balloon, wind_interp.world_bounds),
    )
    return Simulation(resolved, wind_interp, env_idx=env_idx)
