"""Фабрика зависимостей и точка входа для запуска визуализации."""

from dataclasses import replace

import pyvista as pv

from diplom.config import AppConfig, VisualizationConfig
from diplom.world import log_world_bounds
from diplom.wind.factory import build_wind_interpolator
from .balloon_simulation import BalloonSimulation
from .hud import BalloonHUD
from ..sim.factory import create_simulation


class VisualizationRunner:
    """Собирает зависимости (plotter, HUD, частицы) и запускает визуализацию."""

    def build_plotter(self, config: VisualizationConfig) -> pv.Plotter:
        """Создать и настроить PyVista Plotter."""
        plotter = pv.Plotter(window_size=list(config.window_size))
        plotter.set_background(config.bg_bottom, top=config.bg_top)
        return plotter

    @staticmethod
    def build_hud(plotter: pv.Plotter) -> BalloonHUD:
        """Создать HUD для данного плоттера."""
        return BalloonHUD(plotter)

    def run_real(self, config: AppConfig) -> None:
        """Загрузить реальные данные ветра и запустить визуализацию."""
        wind_interpolator = build_wind_interpolator(config.wind)
        log_world_bounds(
            wind_interpolator.world_bounds,
            origin_lat=wind_interpolator.origin_lat,
            origin_lon=wind_interpolator.origin_lon,
            wind_path=config.wind.path,
            prefix="[viz-real]",
        )
        # Визуализация использует общий sim-конфиг, но стартовое время берёт из визуального слоя.
        simulation_config = replace(
            config.simulation,
            balloon=replace(config.simulation.balloon, sim_time=config.visualization.sim_start_time),
        )
        simulation = create_simulation(simulation_config, wind_interpolator)
        plotter = self.build_plotter(config.visualization)
        hud = self.build_hud(plotter)

        BalloonSimulation(
            wind_interpolator=wind_interpolator,
            sim=simulation,
            plotter=plotter,
            hud=hud,
        ).run()
