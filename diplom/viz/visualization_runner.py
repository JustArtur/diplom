"""Фабрика зависимостей и точка входа для запуска визуализации."""

from dataclasses import replace

import pyvista as pv

from diplom.config import AppConfig, VisualizationConfig
from diplom.wind.factory import build_wind_interpolator
from .balloon_simulation import BalloonSimulation
from .hud import BalloonHUD
from ..sim.simulation import Simulation


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
        # Визуализация использует общий sim-конфиг, но стартовое время берёт из визуального слоя.
        simulation_config = replace(
            config.simulation,
            balloon=replace(config.simulation.balloon, sim_time=config.visualization.sim_start_time),
        )
        simulation = Simulation(simulation_config, wind_interpolator)
        plotter = self.build_plotter(config.visualization)
        hud = self.build_hud(plotter)

        BalloonSimulation(
            wind_interpolator=wind_interpolator,
            sim=simulation,
            plotter=plotter,
            hud=hud,
        ).run()
