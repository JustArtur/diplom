from __future__ import annotations

import os
import webbrowser
from multiprocessing import get_context
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from diplom.world import WorldBounds
from diplom.train.trajectory_render_worker import (
    TRAJECTORY_RENDER_SOCKET_ENV,
    TrajectoryRenderRequest,
    cleanup_snapshots_dir,
    snapshot_path_for,
    start_trajectory_render_worker,
    stop_trajectory_render_worker,
    submit_trajectory_render,
    write_trajectory_snapshot,
)
from diplom.train.trajectory_steps_io import (
    EpisodeFileRef,
    cleanup_steps_dir,
)


class TrajectoryVisualizationCallback(BaseCallback):
    """Асинхронно рендерит HTML-файлы траекторий в отдельном процессе.

    Шаги эпизодов пишутся в JSONL в subprocess среды (без ``env_method`` на
    каждом шаге). Главный процесс раз в rollout забирает пути к файлам и
    ставит задачу воркеру рендера.

    Для каждой среды создаётся ``env_XXX.html`` (live-viewer) и ping-pong
    ``env_XXX_d0.js`` / ``env_XXX_d1.js`` в подкаталоге ``_live/``. Откройте HTML
    в браузере — график обновляется через Plotly.react, камера сохраняется
    в localStorage.

    Args:
        output_dir: каталог для HTML-файлов (создаётся автоматически).
        open_in_browser: открыть ``env_XXX.html`` в браузере при старте обучения.
        verbose: 0 — тихо, 1 — печатать сводку после каждого rollout.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        open_in_browser: bool = False,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self._output_dir = Path(output_dir)
        self._open_in_browser = open_in_browser

        self._ctx = get_context("spawn")
        self._render_queue = None
        self._render_process = None
        self._world_bounds: WorldBounds | None = None

    def _on_training_start(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._world_bounds = _get_world_bounds(self.training_env)
        try:
            from diplom.train.cpu_profiling import (
                CPROFILE_DIR_ENV,
                CPROFILE_PROFILE_TRAJECTORY_ENV,
            )
            from diplom.train.memory_profiling import (
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
                )
        except Exception:  # noqa: BLE001
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
        snapshot_path = snapshot_path_for(self._output_dir, request.num_timesteps)
        write_trajectory_snapshot(snapshot_path, request)
        submit_trajectory_render(self._render_queue, snapshot_path)

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
                current_steps_paths[env_idx] = Path(state["current_steps_path"])
                current_step_counts[env_idx] = step_count

        return TrajectoryRenderRequest(
            num_timesteps=int(self.num_timesteps),
            n_envs=int(getattr(self.training_env, "num_envs", 1)),
            episode_counts=episode_counts,
            history=history,
            current_steps_paths=current_steps_paths,
            current_step_counts=current_step_counts,
            world_bounds=self._world_bounds,
        )


def _open_trajectory_viewers(
    output_dir: Path,
    *,
    n_envs: int,
    world_bounds: WorldBounds | None,
) -> None:
    """Создать live-viewer при необходимости и открыть вкладки в браузере."""
    import plotly.graph_objects as go

    from diplom.viz.trajectory_plot import (
        apply_figure_layout,
        compute_trajectory_bounds,
        save_live_figure,
    )

    bounds = compute_trajectory_bounds([], world_bounds=world_bounds)
    for env_idx in range(n_envs):
        html_path = output_dir / f"env_{env_idx:03d}.html"
        if not html_path.exists():
            fig = go.Figure()
            apply_figure_layout(
                fig,
                f"env_{env_idx:03d} · ожидание данных…",
                bounds,
            )
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
