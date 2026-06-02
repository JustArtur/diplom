"""Интерактивная 3D-визуализация стратосферного аэростата в ветровом поле."""

from __future__ import annotations

import time
from functools import partial

import numpy as np
import pyvista as pv

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
    DEFAULT_AIR_PUMP_SPEED,
    MAX_FRAME_DELTA_S,
    MIN_TICK_INTERVAL_S,
    ROPE_BOTTOM_Z,
    ROPE_TOP_Z,
    TARGET_RADIUS,
    TERRAIN_PATCH_SIZE_M,
    VISIBLE_RADIUS,
    WIND_SPEED_MAX_COLOR,
)
from .hud import BalloonHUD, HudState
from .particles import WindParticles
from .terrain import build_bushes_mesh, build_terrain_plane
from diplom.sim.simulation import SimResult, Simulation


class BalloonSimulation:
    """Интерактивная визуализация стратостата в ветровом поле (PyVista).

    Координаты ERA5 в локальных метрах достигают миллионов по X/Y — в VTK/OpenGL
    при таких значениях теряется точность. Все объекты рисуются в системе,
    привязанной к текущей позиции аэростата (аэростат в начале координат).
    """

    def __init__(
        self,
        *,
        wind_interpolator: WindInterpolator,
        plotter: pv.Plotter,
        hud: BalloonHUD,
        sim: Simulation,
    ) -> None:
        self.position = np.asarray(sim.position, dtype=np.float64).copy()
        self.target_position = np.asarray(sim.target_position, dtype=np.float64).copy()
        self.air_pump_speed = 0.0

        self.start_time = time.monotonic()
        self._last_tick = self.start_time

        self.wind_interpolator = wind_interpolator
        self.sim = sim

        self.plotter = plotter
        self._hud = hud
        self._particles = WindParticles(
            self.position.astype(np.float32),
            wind_interpolator,
            sim.sim_time,
        )
        self._balloon_actors: list[pv.Actor] = []
        self._target_actor: pv.Actor | None = None
        self._terrain_actor: pv.Actor | None = None
        self._bushes_actor: pv.Actor | None = None
        self._key_release_observer_id: int | None = None
        self._render_observer_id: int | None = None
        self._interaction_start_observer_id: int | None = None
        self._interaction_end_observer_id: int | None = None
        self._user_camera_active = False

        self._build_scene()
        self._apply_snapshot(self.sim.snapshot())
        self._sync_scene()

    def _scene_origin(self) -> np.ndarray:
        """Мировая позиция аэростата → начало координат сцены."""
        return self.position.copy()

    def _to_scene(self, world: np.ndarray) -> np.ndarray:
        return np.asarray(world, dtype=np.float64) - self._scene_origin()

    def _build_scene(self) -> None:
        self._build_terrain()
        self._build_balloon()
        self._build_target()
        self._init_wind_mesh()
        self._setup_controls()
        self._init_camera()

    def _build_terrain(self) -> None:
        """Локальный зелёный патч с рельефом и кустами."""
        self._terrain_actor = self.plotter.add_mesh(
            build_terrain_plane(),
            scalars="colors",
            rgb=True,
            show_edges=False,
            show_scalar_bar=False,
            name="terrain",
        )
        bushes = build_bushes_mesh()
        if bushes.n_points > 0:
            self._bushes_actor = self.plotter.add_mesh(
                bushes,
                scalars="colors",
                rgb=True,
                show_edges=False,
                show_scalar_bar=False,
                name="bushes",
            )

    def _build_balloon(self) -> None:
        """Аэростат в начале координат сцены."""
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

    def _build_target(self) -> None:
        target = pv.Sphere(radius=TARGET_RADIUS, center=(0, 0, 0))
        self._target_actor = self.plotter.add_mesh(target, color="tomato", name="target")

    def _init_wind_mesh(self) -> None:
        self.plotter.add_mesh(
            self._particles.mesh,
            scalars="speed",
            cmap="coolwarm",
            clim=[0.0, WIND_SPEED_MAX_COLOR],
            line_width=2,
            show_scalar_bar=False,
            name="wind",
        )

    def _apply_snapshot(self, state: SimResult) -> None:
        self.position = np.asarray(state.position, dtype=np.float64)
        self.target_position = np.asarray(state.target_position, dtype=np.float64)
        self.vertical_speed = float(state.vertical_speed)
        self.vertical_acceleration = float(state.vertical_acceleration)
        self.energy_spent = float(state.energy_spent)
        self.temperature = float(state.temperature)
        self.pressure = float(state.pressure)
        self.wind = (
            float(state.wind[0]),
            float(state.wind[1]),
            float(state.wind[2]),
        )

    @staticmethod
    def _set_actor_position(actor: pv.Actor, xyz: tuple[float, float, float]) -> None:
        actor.position = xyz

    def _sync_terrain_position(self) -> None:
        """Уровень земли (z=0 м AMSL) в координатах сцены."""
        ground_z = -float(self.position[2])
        ground_pos = (0.0, 0.0, ground_z)
        if self._terrain_actor is not None:
            self._set_actor_position(self._terrain_actor, ground_pos)
        if self._bushes_actor is not None:
            self._set_actor_position(self._bushes_actor, ground_pos)

    def _move_target_to(self) -> None:
        if self._target_actor is None:
            return
        scene = self._to_scene(self.target_position)
        self._set_actor_position(
            self._target_actor,
            (float(scene[0]), float(scene[1]), float(scene[2])),
        )

    def _sync_target_line(self) -> None:
        start = (0.0, 0.0, 0.0)
        end = self._to_scene(self.target_position)
        line = pv.Line(start, tuple(end))
        self.plotter.add_mesh(
            line,
            color="white",
            line_width=4,
            name="target_line",
        )

    def _sync_hud(self) -> None:
        self._hud.update(
            HudState(
                position=self.position.astype(np.float32).copy(),
                target_position=self.target_position.astype(np.float32),
                energy_spent=self.energy_spent,
                vertical_speed=self.vertical_speed,
                vertical_acceleration=self.vertical_acceleration,
                wind=self.wind,
                temperature=self.temperature,
                pressure=self.pressure,
                start_monotonic=self.start_time,
            )
        )

    def _local_scene_bounds(self) -> tuple[float, float, float, float, float, float]:
        target_scene = self._to_scene(self.target_position)
        pad_xy = max(TERRAIN_PATCH_SIZE_M * 0.55, VISIBLE_RADIUS * 2.0)
        z_vals = [0.0, float(target_scene[2]), -float(self.position[2])]
        z_pad = max(400.0, abs(z_vals[1]) * 0.35 + 300.0)
        x_vals = [0.0, float(target_scene[0])]
        y_vals = [0.0, float(target_scene[1])]
        return (
            min(x_vals) - pad_xy,
            max(x_vals) + pad_xy,
            min(y_vals) - pad_xy,
            max(y_vals) + pad_xy,
            min(z_vals) - z_pad,
            max(z_vals) + z_pad,
        )

    def _reset_camera_local(self) -> None:
        self.plotter.reset_camera(bounds=self._local_scene_bounds(), render=False)
        cam = self.plotter.camera
        cam.focal_point = (0.0, 0.0, 0.0)
        direction = np.asarray(CAMERA_INITIAL_OFFSET, dtype=np.float64)
        direction /= np.linalg.norm(direction)
        cam.position = tuple(direction * CAMERA_ORBIT_RADIUS)
        cam.up = CAMERA_INITIAL_VIEW_UP
        target_z = float(self._to_scene(self.target_position)[2])
        cam.clipping_range = (
            max(1.0, CAMERA_ORBIT_RADIUS * 0.05),
            max(CAMERA_ORBIT_RADIUS * 4.0, target_z + 2_000.0),
        )

    def _init_camera(self) -> None:
        self.plotter.add_axes()
        self._reset_camera_local()

    def _follow_camera(self) -> None:
        if self._user_camera_active:
            return
        cam = self.plotter.camera
        direction = (
            np.asarray(cam.position, dtype=np.float64)
            - np.asarray(cam.focal_point, dtype=np.float64)
        )
        dist = float(np.linalg.norm(direction))
        if dist < CAMERA_DIRECTION_EPS:
            direction = CAMERA_INITIAL_OFFSET / CAMERA_ORBIT_RADIUS
        else:
            direction /= dist
        cam.focal_point = (0.0, 0.0, 0.0)
        cam.position = tuple(direction * CAMERA_ORBIT_RADIUS)

    def _sync_scene(self) -> None:
        self._sync_terrain_position()
        self._move_target_to()
        self._sync_target_line()
        self._sync_hud()

    def _setup_controls(self) -> None:
        self.plotter.add_key_event("a", partial(self._set_air_pump_speed, -DEFAULT_AIR_PUMP_SPEED))
        self.plotter.add_key_event("z", partial(self._set_air_pump_speed, DEFAULT_AIR_PUMP_SPEED))
        self.plotter.add_timer_event(
            max_steps=10**9,
            duration=ANIM_INTERVAL_MS,
            callback=self._on_timer,
        )

    def _register_interactor_observers(self) -> None:
        """Подписки на interactor: отпускание клавиш, рендер, ручная камера."""
        iren = self.plotter.iren
        if iren is None:
            return
        if self._key_release_observer_id is None:
            self._key_release_observer_id = iren.add_observer(
                "KeyReleaseEvent",
                self._stop_ballon,
            )
        if self._render_observer_id is None:
            self._render_observer_id = iren.add_observer(
                "RenderEvent",
                self._on_render,
            )
        if self._interaction_start_observer_id is None:
            self._interaction_start_observer_id = iren.add_observer(
                "StartInteractionEvent",
                self._on_start_interaction,
            )
        if self._interaction_end_observer_id is None:
            self._interaction_end_observer_id = iren.add_observer(
                "EndInteractionEvent",
                self._on_end_interaction,
            )

    def _on_start_interaction(self, *_args) -> None:
        self._user_camera_active = True

    def _on_end_interaction(self, *_args) -> None:
        self._user_camera_active = False

    def _set_air_pump_speed(self, speed: float) -> None:
        self.air_pump_speed = speed

    def _stop_ballon(self, interactor, _event) -> None:
        key = interactor.GetKeySym().lower()
        if key in ("a", "z"):
            self.air_pump_speed = 0.0

    def _on_timer(self, *_args) -> None:
        self._do_tick()

    def _on_render(self, *_args) -> None:
        """Второй цикл обновления: анимация не замирает при вращении камеры."""
        self._do_tick(emit_render=False)

    def _do_tick(self, *, emit_render: bool = True) -> None:
        now = time.monotonic()
        dt = now - self._last_tick
        if dt < MIN_TICK_INTERVAL_S:
            return
        self._last_tick = now
        dt = min(dt, MAX_FRAME_DELTA_S)

        self._do_simulation(dt)
        self._particles.step(self.position.astype(np.float32), self.sim.sim_time)

        self._sync_scene()
        self._follow_camera()
        if emit_render:
            self.plotter.render()

    def _do_simulation(self, dt: float) -> None:
        state = self.sim.step(dt, self.air_pump_speed)
        self._apply_snapshot(state)

    def run(self) -> None:
        self._register_interactor_observers()
        self._reset_camera_local()
        self.plotter.show()
