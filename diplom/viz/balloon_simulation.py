"""Интерактивная 3D-визуализация стратосферного аэростата в ветровом поле."""

import time
from functools import partial

import numpy as np
import pyvista as pv

from diplom.shared_constants import WORLD_SIZE
from diplom.wind.interp import WindInterpolator

from .constants import (
    ANIM_INTERVAL_MS,
    BALLOON_RADIUS,
    BASKET_OFFSET_Z,
    BASKET_SIZE,
    CAMERA_DIRECTION_EPS,
    CAMERA_INITIAL_OFFSET,
    CAMERA_INITIAL_VIEW_UP,
    CAMERA_ORBIT_RADIUS,
    MAX_FRAME_DELTA_S,
    DEFAULT_AIR_PUMP_SPEED,
    MIN_TICK_INTERVAL_S,
    ROPE_BOTTOM_Z,
    ROPE_TOP_Z,
    TARGET_RADIUS,
    TERRAIN_AMP_COS,
    TERRAIN_AMP_SIN,
    TERRAIN_FREQ_COS,
    TERRAIN_FREQ_SIN,
    TERRAIN_RESOLUTION,
    WIND_SPEED_MAX_COLOR,
)
from .hud import BalloonHUD, HudState
from .particles import WindParticles
from ..sim.simulation import Simulation


class BalloonSimulation:
    """Интерактивная визуализация стратостата в ветровом поле (PyVista).

    Обязанности: построение сцены, анимационный цикл, физика движения,
    управление камерой и HUD.
    """

    # ──────────────────── Инициализация ────────────────────
    def __init__(self, *, wind_interpolator: WindInterpolator, plotter: pv.Plotter, hud: BalloonHUD,
                 sim: Simulation) -> None:
        # ── Физическое состояние ──
        self.position = sim.position.copy()
        self.target_position = sim.target_position.copy()
        self.air_pump_speed = 0.0

        # ── Время ──
        self.start_time = time.monotonic()
        self.sim_time = sim.sim_time
        self._last_tick = self.start_time

        # ── Ветер ──
        self.wind_interpolator = wind_interpolator
        self.sim = sim

        # ── Визуальные компоненты ──
        self.plotter = plotter
        self._hud = hud
        self._particles = WindParticles(self.position.copy(), wind_interpolator, self.sim_time)

        self._build_scene()

    # ──────────────────── Построение сцены ────────────────────

    def _build_scene(self) -> None:
        """Собрать все элементы 3D-сцены и запустить управление."""
        self._build_terrain()
        self._build_balloon()
        self._build_target()
        self._init_wind_mesh()
        self._setup_controls()
        self._init_camera()

    def _build_terrain(self) -> None:
        """Зелёная поверхность земли с лёгким синтетическим рельефом."""
        plane = pv.Plane(
            center=(0, 0, 0),
            i_size=WORLD_SIZE,
            j_size=WORLD_SIZE,
            i_resolution=TERRAIN_RESOLUTION,
            j_resolution=TERRAIN_RESOLUTION,
        )
        x_pts, y_pts = plane.points[:, 0], plane.points[:, 1]
        plane.points[:, 2] = (
                TERRAIN_AMP_SIN * np.sin(y_pts * TERRAIN_FREQ_SIN)
                + TERRAIN_AMP_COS * np.cos(x_pts * TERRAIN_FREQ_COS)  # Добавляем искусственные неровности
        )
        green = np.array([34, 139, 34], dtype=np.uint8)
        plane.point_data["colors"] = np.tile(green, (plane.n_points, 1))  # Зеленый цвет
        self.plotter.add_mesh(
            plane, scalars="colors", rgb=True, show_edges=False, show_scalar_bar=False,
        )

    def _build_balloon(self) -> None:
        """Создать меши аэростата (оболочка + верёвка + корзина) один раз."""
        sphere = pv.Sphere(radius=BALLOON_RADIUS, center=(0, 0, 0))
        rope = pv.Line((0, 0, ROPE_TOP_Z), (0, 0, ROPE_BOTTOM_Z))
        basket = pv.Cube(
            center=(0, 0, ROPE_BOTTOM_Z + BASKET_OFFSET_Z),
            x_length=BASKET_SIZE[0],
            y_length=BASKET_SIZE[1],
            z_length=BASKET_SIZE[2],
        )
        self._balloon_actors = [
            self.plotter.add_mesh(sphere, color="gold", name="balloon"),
            self.plotter.add_mesh(rope, color="saddlebrown", line_width=3, name="rope"),
            self.plotter.add_mesh(basket, color="sienna", name="basket"),
        ]
        self._move_balloon_to()

    def _build_target(self) -> None:
        """Маркер цели и линия «аэростат → цель»."""
        target = pv.Sphere(radius=TARGET_RADIUS, center=tuple(self.target_position))
        self.plotter.add_mesh(target, color="tomato", name="target")
        self._sync_target_line()

    def _init_wind_mesh(self) -> None:
        """Добавить меш ветровых частиц в сцену (меш создаётся внутри WindParticles)."""
        self.plotter.add_mesh(
            self._particles.mesh,
            scalars="speed",
            cmap="coolwarm",
            clim=[0.0, WIND_SPEED_MAX_COLOR],
            line_width=2,
            show_scalar_bar=False,
            name="wind",
        )

    # ──────────────────── Обновление визуалов ────────────────────

    def _move_balloon_to(self) -> None:
        """Переместить акторы аэростата через VTK SetPosition."""
        for actor in self._balloon_actors:
            actor.position = self.position

    def _sync_target_line(self) -> None:
        """Обновить линию «аэростат → цель»."""
        line = pv.Line(tuple(self.position), tuple(self.target_position))
        self.plotter.add_mesh(line, color="white", line_width=2, name="target_line")

    def _sync_hud(self) -> None:
        """Передать текущее состояние в HUD."""
        self._hud.update(HudState(
            position=self.position.copy(),
            target_position=self.target_position,
            energy_spent=self.energy_spent,
            vertical_speed=self.vertical_speed,
            vertical_acceleration=self.vertical_acceleration,
            wind=self.wind,
            temperature=self.temperature,
            pressure=self.pressure,
            start_monotonic=self.start_time,
        ))

    # ──────────────────── Камера ────────────────────

    def _init_camera(self) -> None:
        """Начальная позиция камеры."""
        self.plotter.add_axes()
        cam = self.plotter.camera
        cam.focal_point = tuple(self.position)
        cam.position = tuple(self.position + CAMERA_INITIAL_OFFSET)
        cam.up = CAMERA_INITIAL_VIEW_UP

    def _follow_camera(self) -> None:
        """Переместить камеру за аэростатом, сохраняя направление взгляда."""
        cam = self.plotter.camera
        direction = np.asarray(cam.position, dtype=np.float32) - np.asarray(cam.focal_point, dtype=np.float32)

        dist = float(np.linalg.norm(direction))
        # Нормализуем вектор
        if dist < CAMERA_DIRECTION_EPS:
            direction = CAMERA_INITIAL_OFFSET / CAMERA_ORBIT_RADIUS
        else:
            direction /= dist
        cam.focal_point = tuple(self.position)
        cam.position = tuple(self.position + direction * CAMERA_ORBIT_RADIUS)

    # ──────────────────── Управление ────────────────────

    def _setup_controls(self) -> None:
        """Привязать клавиши и таймеры анимации."""
        self.plotter.add_key_event("a", partial(self._set_air_pump_speed, -DEFAULT_AIR_PUMP_SPEED))
        self.plotter.add_key_event("z", partial(self._set_air_pump_speed, DEFAULT_AIR_PUMP_SPEED))
        self.plotter.iren.add_observer("KeyReleaseEvent", self._stop_ballon)

        self.plotter.add_timer_event(max_steps=10 ** 9, duration=ANIM_INTERVAL_MS, callback=self._on_timer)
        self.plotter.iren.add_observer("RenderEvent", self._on_render)

    def _set_air_pump_speed(self, speed: float) -> None:
        """Задаем скорость закачки воздуха в баллон."""
        self.air_pump_speed = speed

    def _stop_ballon(self, obj, _event) -> None:
        """Убираем скорость шара"""
        key = obj.GetKeySym().lower()
        if key in ("a", "z"):
            self.air_pump_speed = 0.0

    # ──────────────────── Игровой цикл ────────────────────

    def _on_timer(self, _step: int = 0) -> None:
        self._do_tick()

    def _on_render(self, *_args) -> None:
        self._do_tick()

    def _do_tick(self) -> None:
        """Один шаг симуляции + обновление визуализации."""
        now = time.monotonic()
        dt = now - self._last_tick
        if dt < MIN_TICK_INTERVAL_S:
            return
        self._last_tick = now
        dt = min(dt, MAX_FRAME_DELTA_S)

        # ── Симуляция ──
        self._adjust_sim_time(dt)
        self._do_simulation(dt)
        self._particles.step(self.position, self.sim_time)


        # ── Визуалы ──
        self._move_balloon_to()
        self._follow_camera()
        self._sync_target_line()
        self._sync_hud()
        self.plotter.renderer.ResetCameraClippingRange()

    def _do_simulation(self, dt):
        state = self.sim.step(dt, self.air_pump_speed)

        self.position = state.position
        self.vertical_speed = state.vertical_speed
        self.vertical_acceleration = state.vertical_acceleration
        self.energy_spent = state.energy_spent
        self.temperature = state.temperature
        self.pressure = state.pressure
        self.wind = state.wind
    # ──────────────────── Физика ────────────────────

    def _adjust_sim_time(self, dt: float) -> None:
        """Продвинуть время симуляции для запросов к временным слоям ветра."""
        self.sim_time += np.timedelta64(int(dt * 1000), "ms")

    # ──────────────────── Запуск ────────────────────

    def run(self) -> None:
        """Открыть окно и запустить главный цикл PyVista."""
        self.plotter.show()
