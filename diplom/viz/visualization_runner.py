"""Фабрика зависимостей и точка входа для запуска визуализации."""

from pathlib import Path

import numpy as np
import pyvista as pv

from diplom.wind.interp import WindInterpolator
from .balloon_simulation import BalloonSimulation
from .constants import WINDOW_SIZE
from .hud import BalloonHUD
from ..sim.simulation import Simulation, SimParams


class VisualizationRunner:
    """Собирает зависимости (plotter, HUD, частицы) и запускает визуализацию."""

    def __init__(self, *, window_size: tuple[int, int] = WINDOW_SIZE, bg_bottom: str = "deepskyblue",
                 bg_top: str = "midnightblue", ) -> None:
        self.window_size = window_size  # размер окна (ш × в, пикс.)
        self.bg_bottom = bg_bottom  # цвет нижней части фона (горизонт)
        self.bg_top = bg_top  # цвет верхней части фона (зенит)

    def build_plotter(self) -> pv.Plotter:
        """Создать и настроить PyVista Plotter."""
        plotter = pv.Plotter(window_size=list(self.window_size))
        plotter.set_background(self.bg_bottom, top=self.bg_top)
        return plotter

    @staticmethod
    def build_hud(plotter: pv.Plotter) -> BalloonHUD:
        """Создать HUD для данного плоттера."""
        return BalloonHUD(plotter)

    def run_real(self, *, data_path: Path, origin_lat: float, origin_lon: float, start_time: np.datetime64) -> None:
        """Загрузить реальные данные ветра и запустить визуализацию."""
        wind_interpolator = WindInterpolator.from_file(path=data_path, origin_lat=origin_lat, origin_lon=origin_lon)
        simulation = Simulation(SimParams(wind_interp=wind_interpolator))
        plotter = self.build_plotter()
        hud = self.build_hud(plotter)

        BalloonSimulation(
            wind_interpolator=wind_interpolator,
            sim=simulation,
            plotter=plotter,
            hud=hud,
            sim_start_time=start_time,
        ).run()
