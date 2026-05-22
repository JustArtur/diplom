from __future__ import annotations

import re
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Sequence, Union

_RUN_DIR_RE = re.compile(r"^PPO_(\d+)$", re.IGNORECASE)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from diplom.config import AppConfig, EnvironmentConfig, WindConfig
from diplom.envs.balloon_env import BalloonEnv
from diplom.envs.factory import build_env
from diplom.torch_device import resolve_torch_device
from diplom.rl.callbacks.clip_log_std import ClipLogStdCallback
from diplom.rl.callbacks.curriculum import (
    EntCoefScheduleCallback,
    TrainEpisodeLengthCurriculumCallback,
    initial_episode_length_curriculum_max_steps,
)
from diplom.rl.callbacks.episode_stats import EpisodeStatsCallback
from diplom.rl.callbacks.info_logging import InfoLoggingCallback
from diplom.rl.ppo.models import get_model_spec
from diplom.rl.ppo.policy import build_ppo_policy_kwargs
from diplom.world import log_world_bounds
from diplom.wind.factory import build_wind_interpolator
from diplom.wind.interp import ensure_wind_interpolator_cache

def _env_factory(
    env_config: EnvironmentConfig,
    wind_config: WindConfig,
    env_idx: int | None = None,
) -> BalloonEnv:
    """Создаёт одну среду внутри рабочего подпроцесса.

    Функция должна быть определена на уровне модуля (не lambda и не closure),
    чтобы pickle мог сериализовать её при передаче в SubprocVecEnv.
    """
    from diplom.dev.profiling.cpu import start_process_cprofile_if_enabled
    from diplom.dev.profiling.memory import env_process_name, start_process_memray_if_enabled

    process_name = env_process_name(env_idx)
    start_process_memray_if_enabled(process_name)
    start_process_cprofile_if_enabled(process_name)
    env = build_env(env_config, wind_config, env_idx=env_idx)
    return Monitor(env)


def _make_vec_env(
    config: AppConfig,
    *,
    force_dummy: bool = False,
    trajectory_steps_dir: Path | None = None,
    model_name: str = "default",
) -> VecEnv:
    n_envs = max(1, config.training.n_envs)
    ensure_wind_interpolator_cache(config.wind.path)
    train_max_steps = config.environment.train_max_episode_steps
    if config.training.episode_length_curriculum_enabled:
        train_max_steps = initial_episode_length_curriculum_max_steps(
            config.training.episode_length_curriculum_stages
        )
    env_config = replace(
        config.environment,
        max_episode_steps=train_max_steps,
        trajectory_steps_dir=trajectory_steps_dir,
    )
    factories = [
        partial(_env_factory, env_config=env_config, wind_config=config.wind, env_idx=env_idx)
        for env_idx in range(n_envs)
    ]
    if force_dummy or n_envs == 1:
        return DummyVecEnv(factories)

    model_spec = get_model_spec(model_name)
    if model_spec.recurrent:
        from stable_baselines3.common.vec_env import SubprocVecEnv

        return SubprocVecEnv(factories, start_method="spawn")

    from diplom.rl.vec_env.policy_shmem_rollout import PolicyShmemSubprocVecEnv

    return PolicyShmemSubprocVecEnv(
        factories,
        rollout_steps=config.training.ppo_n_steps,
        model_name=model_name,
        start_method="spawn",
    )


def _ppo_index_from_dir_name(name: str) -> int | None:
    match = _RUN_DIR_RE.fullmatch(name)
    if match is None:
        return None
    return int(match.group(1))


def _next_ppo_index(logdir: Path) -> int:
    nums: list[int] = []
    for path in logdir.iterdir():
        if not path.is_dir():
            continue
        idx = _ppo_index_from_dir_name(path.name)
        if idx is not None:
            nums.append(idx)
    return max(nums) + 1 if nums else 0


def _latest_run_dir(logdir: Path) -> Path | None:
    best: tuple[int, Path] | None = None
    for path in logdir.iterdir():
        if not path.is_dir():
            continue
        idx = _ppo_index_from_dir_name(path.name)
        if idx is None:
            continue
        if best is None or idx > best[0]:
            best = (idx, path)
    return best[1] if best is not None else None


def training_run_dir(
    logdir: Path,
    *,
    reuse_latest: bool = False,
) -> Path:
    """Каталог run-а: ``PPO_N`` внутри logdir."""
    logdir.mkdir(parents=True, exist_ok=True)

    if reuse_latest:
        existing = _latest_run_dir(logdir)
        if existing is not None:
            return existing

    index = _next_ppo_index(logdir)
    run_dir = logdir / f"PPO_{index}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def train_ppo(
    config: AppConfig,
    callbacks: Sequence[BaseCallback] | None = None,
    *,
    force_dummy_vec_env: bool = False,
    run_dir: Path | None = None,
    enable_trajectory_viz: bool = True,
    open_trajectory_viz: bool = False,
    resume: bool = False,
) -> Path:
    """Train PPO agent on the balloon environment.

    Returns:
        Каталог текущего run-а (TensorBoard ``tb_*``, подкаталог ``trajectories``).
    """
    from diplom.trajectory.live.callback import TrajectoryVisualizationCallback

    logdir = config.training.logdir
    logdir.mkdir(parents=True, exist_ok=True)
    device = resolve_torch_device(config.training.device)
    ppo_verbose = config.training.verbose
    model_spec = get_model_spec(config.training.model_name)
    print(f"[train_ppo] Using device={device}")  # noqa: T201
    print(f"[train_ppo] PPO model: {model_spec.name} ({model_spec.policy_type})")  # noqa: T201
    print(f"[train_ppo] Reward: {config.environment.reward_name}")  # noqa: T201
    print(f"[train_ppo] Obs: {config.environment.obs_name}")  # noqa: T201
    print(f"[train_ppo] Dataset: {config.wind.path.name}")  # noqa: T201
    print(f"[train_ppo] Log directory: {logdir}")  # noqa: T201

    model_path = logdir / "ppo_model.zip"
    model_exists = model_path.exists()
    continuing = resume and model_exists
    if run_dir is None:
        run_dir = training_run_dir(logdir, reuse_latest=continuing)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)

    reset_num_timesteps = not continuing
    if continuing:
        print(f"[train_ppo] Продолжаем run и TensorBoard в {run_dir}")  # noqa: T201
    traj_dir = run_dir / "trajectories"
    print(f"[train_ppo] Run directory: {run_dir}")  # noqa: T201

    trajectory_steps_dir = traj_dir if enable_trajectory_viz else None
    if enable_trajectory_viz:
        print(f"[train_ppo] Trajectory viz: {traj_dir.resolve()}")  # noqa: T201
    else:
        print("[train_ppo] Trajectory viz: off (--no-trajectories)")  # noqa: T201
    vec_env = _make_vec_env(
        config,
        force_dummy=force_dummy_vec_env,
        trajectory_steps_dir=trajectory_steps_dir,
        model_name=config.training.model_name,
    )
    if model_spec.recurrent and not force_dummy_vec_env and config.training.n_envs > 1:
        print("[train_ppo] VecEnv: SubprocVecEnv (RecurrentPPO, без worker rollout)")  # noqa: T201
    elif not force_dummy_vec_env and config.training.n_envs > 1:
        print("[train_ppo] VecEnv: PolicyShmemSubprocVecEnv (worker policy + shmem)")  # noqa: T201
    info_callback = InfoLoggingCallback()
    episode_stats_callback = EpisodeStatsCallback()

    probe_interp = build_wind_interpolator(config.wind)
    try:
        log_world_bounds(
            probe_interp.world_bounds,
            origin_lat=probe_interp.origin_lat,
            origin_lon=probe_interp.origin_lon,
            wind_path=config.wind.path,
            prefix="[train_ppo]",
        )
    finally:
        probe_interp.close()

    traj_callback = (
        TrajectoryVisualizationCallback(
            output_dir=traj_dir,
            wind_dataset_path=config.wind.path,
            open_in_browser=open_trajectory_viz,
        )
        if enable_trajectory_viz
        else None
    )

    use_worker_rollout = (
        not force_dummy_vec_env
        and config.training.n_envs > 1
        and not model_spec.recurrent
    )
    if model_spec.recurrent:
        from sb3_contrib import RecurrentPPO

        ppo_cls: type[PPO] = RecurrentPPO
    elif use_worker_rollout:
        from diplom.rl.ppo.worker_rollout import WorkerRolloutPPO

        ppo_cls = WorkerRolloutPPO
    else:
        ppo_cls = PPO

    model: PPO

    def _save_model() -> None:
        model.save(logdir / "ppo_model")
        print(f"[train_ppo] Модель сохранена в {model_path}")  # noqa: T201

    try:
        # Если в logdir уже есть сохранённая модель, продолжаем обучение с неё.
        # Иначе инициализируем новую.
        if model_path.exists() and resume:
            print(f"[train_ppo] Продолжаем обучение из {model_path}")  # noqa: T201
            model = ppo_cls.load(model_path, env=vec_env, device=device)
            model.tensorboard_log = str(run_dir)
        else:
            if model_path.exists() and not resume:
                print(  # noqa: T201
                    f"[train_ppo] {model_path} найден, но --resume не задан — "
                    "обучаем новую модель (старый файл будет перезаписан в конце)"
                )
            model = ppo_cls(
                policy=model_spec.policy_type,
                env=vec_env,
                # Собираем больше опыта перед каждым обновлением — лучше
                # использование данных и меньше накладных расходов обновления.
                n_steps=config.training.ppo_n_steps,
                # 512 делит rollout_size=4096*n_envs нацело и эффективнее на MPS/CPU.
                batch_size=512,
                n_epochs=10,
                learning_rate=config.training.learning_rate,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=config.training.ent_coef_end,
                vf_coef=0.5,
                max_grad_norm=config.training.max_grad_norm,
                policy_kwargs=build_ppo_policy_kwargs(config.training.model_name),
                verbose=ppo_verbose,
                tensorboard_log=str(run_dir),
                seed=config.training.seed,
                device=device,
            )
        model.verbose = ppo_verbose
        extra_callbacks = list(callbacks) if callbacks is not None else []
        learn_callbacks: list[BaseCallback] = [
            info_callback,
            episode_stats_callback,
            ClipLogStdCallback(model_name=config.training.model_name),
            EntCoefScheduleCallback(
                start=config.training.ent_coef_start,
                end=config.training.ent_coef_end,
                decay_timesteps=config.training.ent_coef_decay_timesteps,
            ),
        ]
        if config.training.episode_length_curriculum_enabled:
            learn_callbacks.append(
                TrainEpisodeLengthCurriculumCallback(
                    stages=config.training.episode_length_curriculum_stages,
                    verbose=max(0, ppo_verbose - 1),
                )
            )
        if traj_callback is not None:
            learn_callbacks.append(traj_callback)
        learn_callbacks.extend(extra_callbacks)
        try:
            model.learn(
                total_timesteps=config.training.total_timesteps,
                callback=CallbackList(learn_callbacks),
                reset_num_timesteps=reset_num_timesteps,
                tb_log_name="tb",
            )
        except KeyboardInterrupt:
            _save_model()
            raise
        else:
            _save_model()
    finally:
        vec_env.close()

    return run_dir
