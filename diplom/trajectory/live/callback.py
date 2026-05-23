from __future__ import annotations

import os
import traceback
import webbrowser
from multiprocessing import get_context
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from diplom.world import WorldBounds
from diplom.trajectory.live.render_worker import (
    COMBINED_TRAJECTORY_HTML,
    TRAJECTORY_RENDER_SOCKET_ENV,
    TrajectoryRenderRequest,
    cleanup_snapshots_dir,
    render_queue_id,
    snapshot_path_for,
    start_trajectory_render_worker,
    stop_trajectory_render_worker,
    submit_trajectory_render,
    write_trajectory_snapshot,
)
from diplom.trajectory.steps_io import (
    EpisodeFileRef,
    cleanup_steps_dir,
)


class TrajectoryVisualizationCallback(BaseCallback):
    """Асинхронно рендерит HTML-файлы траекторий в отдельном процессе.

    Шаги эпизодов пишутся в JSONL в subprocess среды (без ``env_method`` на
    каждом шаге). Главный процесс раз в rollout забирает пути к файлам и
    ставит задачу воркеру рендера.

    По умолчанию (``combined_html=True``) все среды одного обучения рендерятся
    в один ``trajectories.html``. При ``combined_html=False`` — отдельный
    ``env_XXX.html`` на каждый env-процесс (история + текущий эпизод).

    Args:
        output_dir: каталог для HTML-файлов (создаётся автоматически).
        combined_html: один HTML на всё обучение или отдельный файл на env-процесс.
        open_in_browser: открыть viewer в браузере при старте обучения.
        verbose: 0 — тихо, 1 — печатать сводку после каждого rollout.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        wind_dataset_path: Path | None = None,
        show_wind_cones: bool = False,
        combined_html: bool = True,
        open_in_browser: bool = False,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self._output_dir = Path(output_dir)
        self._wind_dataset_path = Path(wind_dataset_path) if wind_dataset_path is not None else None
        self._show_wind_cones = show_wind_cones
        self._combined_html = combined_html
        self._open_in_browser = open_in_browser

        self._ctx = get_context("spawn")
        self._render_queue = None
        self._render_process = None
        self._world_bounds: WorldBounds | None = None

    def _on_training_start(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._world_bounds = _get_world_bounds(self.training_env)
        try:
            from diplom.dev.profiling.cpu import (
                CPROFILE_DIR_ENV,
                CPROFILE_PROFILE_TRAJECTORY_ENV,
            )
            from diplom.dev.profiling.memory import (
                MEMRAY_DIR_ENV,
                MEMRAY_PROFILE_TRAJECTORY_ENV,
            )

            profile_trajectory = (
                os.environ.get(MEMRAY_DIR_ENV) is not None
                and os.environ.get(MEMRAY_PROFILE_TRAJECTORY_ENV) == "1"
            ) or (
                os.environ.get(CPROFILE_DIR_ENV) is not None
                and os.environ.get(CPROFILE_PROFILE_TRAJECTORY_ENV) == "1"
            )
            shared_socket = os.environ.get(TRAJECTORY_RENDER_SOCKET_ENV)
            if shared_socket and not profile_trajectory:
                self._render_queue = shared_socket
                self._render_process = None
            else:
                self._render_queue, self._render_process = start_trajectory_render_worker(
                    ctx=self._ctx,
                    output_dir=self._output_dir,
                    daemon=not profile_trajectory,
                )
            if self._open_in_browser:
                _open_trajectory_viewers(
                    self._output_dir,
                    n_envs=int(getattr(self.training_env, "num_envs", 1)),
                    world_bounds=self._world_bounds,
                    combined_html=self._combined_html,
                )
            if self.verbose:
                mode = "shared socket" if isinstance(self._render_queue, str) else "local worker"
                print(  # noqa: T201
                    f"[trajectory_viz] рендер включён ({mode}) → {self._output_dir.resolve()}"
                )
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            self._render_queue = None
            self._render_process = None

    def _on_step(self) -> bool:
        # SB3 требует реализацию; шаги пишутся в subprocess (BalloonEnv).
        return True

    def _on_training_end(self) -> None:
        stop_trajectory_render_worker(self._render_queue, self._render_process)
        self._render_queue = None
        self._render_process = None
        cleanup_snapshots_dir(self._output_dir)
        cleanup_steps_dir(self._output_dir)

    def _on_rollout_end(self) -> None:
        """Собрать пути к JSONL из subprocess и передать воркеру рендера."""
        if self._render_queue is None:
            return

        request = self._build_render_request()
        if self._combined_html:
            snapshot_path = snapshot_path_for(self._output_dir, request.num_timesteps)
            write_trajectory_snapshot(snapshot_path, request)
            submit_trajectory_render(
                self._render_queue,
                snapshot_path,
                queue_id=render_queue_id(self._output_dir),
            )
            return

        for env_idx in sorted(
            set(request.history) | set(request.current_steps_paths)
        ):
            env_request = _subset_render_request(request, env_idx)
            snapshot_path = snapshot_path_for(
                self._output_dir,
                request.num_timesteps,
                env_idx,
            )
            write_trajectory_snapshot(snapshot_path, env_request)
            submit_trajectory_render(
                self._render_queue,
                snapshot_path,
                queue_id=render_queue_id(self._output_dir, env_idx),
            )

    def _build_render_request(self) -> TrajectoryRenderRequest:
        states = self.training_env.env_method("get_trajectory_viz_state")
        episode_counts: dict[int, int] = {}
        history: dict[int, list[EpisodeFileRef]] = {}
        current_steps_paths: dict[int, Path] = {}
        current_step_counts: dict[int, int] = {}

        for state in states:
            if not state:
                continue
            env_idx = int(state["env_idx"])
            episode_counts[env_idx] = int(state["episode_count"])
            history[env_idx] = list(state["history"])
            step_count = int(state["current_step_count"])
            if step_count > 0:
                current_steps_paths[env_idx] = Path(state["current_steps_path"]).resolve()
                current_step_counts[env_idx] = step_count

        return TrajectoryRenderRequest(
            num_timesteps=int(self.num_timesteps),
            n_envs=int(getattr(self.training_env, "num_envs", 1)),
            episode_counts=episode_counts,
            history=history,
            current_steps_paths=current_steps_paths,
            current_step_counts=current_step_counts,
            world_bounds=self._world_bounds,
            wind_dataset_path=self._wind_dataset_path,
            show_wind_cones=self._show_wind_cones,
            combined_html=self._combined_html,
        )


def _subset_render_request(
    request: TrajectoryRenderRequest,
    env_idx: int,
) -> TrajectoryRenderRequest:
    return TrajectoryRenderRequest(
        num_timesteps=request.num_timesteps,
        n_envs=1,
        episode_counts={env_idx: request.episode_counts.get(env_idx, 0)},
        history={env_idx: list(request.history.get(env_idx, []))},
        current_steps_paths={
            idx: path
            for idx, path in request.current_steps_paths.items()
            if idx == env_idx
        },
        current_step_counts={
            idx: count
            for idx, count in request.current_step_counts.items()
            if idx == env_idx
        },
        world_bounds=request.world_bounds,
        wind_dataset_path=request.wind_dataset_path,
        show_wind_cones=request.show_wind_cones,
        combined_html=request.combined_html,
    )


def _open_trajectory_viewers(
    output_dir: Path,
    *,
    n_envs: int,
    world_bounds: WorldBounds | None,
    combined_html: bool = True,
) -> None:
    """Создать live-viewer при необходимости и открыть вкладки в браузере."""
    from diplom.viz.plotly.episode_figure import (
        build_placeholder_combined_figure,
        build_placeholder_live_figure,
    )
    from diplom.viz.plotly.trajectory import save_live_figure

    if combined_html:
        html_path = output_dir / COMBINED_TRAJECTORY_HTML
        if not html_path.exists():
            fig = build_placeholder_combined_figure(n_envs, world_bounds)
            save_live_figure(fig, html_path, generation=0)
        webbrowser.open(html_path.resolve().as_uri())
        return

    for env_idx in range(n_envs):
        html_path = output_dir / f"env_{env_idx:03d}.html"
        if not html_path.exists():
            fig = build_placeholder_live_figure(env_idx, world_bounds)
            save_live_figure(fig, html_path, generation=0)
        webbrowser.open(html_path.resolve().as_uri())


def _get_world_bounds(training_env) -> WorldBounds | None:
    """Попробовать вытащить общие границы мира из VecEnv."""
    if training_env is None:
        return None

    try:
        bounds_list = training_env.get_attr("world_bounds")
    except Exception:  # noqa: BLE001
        return None

    if not bounds_list:
        return None

    bounds = bounds_list[0]
    return bounds if isinstance(bounds, WorldBounds) else None
