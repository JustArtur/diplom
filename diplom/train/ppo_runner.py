from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Sequence, Union

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from diplom.config import AppConfig, EnvironmentConfig, WindConfig
from diplom.envs.balloon_env import BalloonEnv
from diplom.envs.factory import build_env
from diplom.torch_device import resolve_torch_device
from diplom.train.info_logging_callback import InfoLoggingCallback
from diplom.train.run_dirs import next_run_dir
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
    from diplom.train.cpu_profiling import start_process_cprofile_if_enabled
    from diplom.train.memory_profiling import env_process_name, start_process_memray_if_enabled

    process_name = env_process_name(env_idx)
    start_process_memray_if_enabled(process_name)
    start_process_cprofile_if_enabled(process_name)
    return build_env(env_config, wind_config, env_idx=env_idx)


def _make_vec_env(
    config: AppConfig,
    *,
    force_dummy: bool = False,
    trajectory_steps_dir: Path | None = None,
) -> VecEnv:
    n_envs = max(1, config.training.n_envs)
    ensure_wind_interpolator_cache(config.wind.path)
    # Включаем рандомизацию старта — нужна при обучении.
    env_config = replace(
        config.environment,
        randomize_start_state=True,
        randomize_start_time=True,
        trajectory_steps_dir=trajectory_steps_dir,
    )
    factories = [
        partial(_env_factory, env_config=env_config, wind_config=config.wind, env_idx=env_idx)
        for env_idx in range(n_envs)
    ]
    # partial — picklable, в отличие от lambda; именно это позволяет SubprocVecEnv
    # передавать фабрику через pickle в дочерний процесс.
    if force_dummy or n_envs == 1:
        return DummyVecEnv(factories)

    if config.training.use_worker_policy_rollout:
        from diplom.train.policy_shmem_rollout_vec_env import PolicyShmemSubprocVecEnv

        return PolicyShmemSubprocVecEnv(
            factories,
            rollout_steps=config.training.ppo_n_steps,
            start_method="spawn",
        )

    # Режим отладки (--main-policy-rollout): policy в main, shmem на каждый step.
    from diplom.train.shmem_vec_env import ShmemSubprocVecEnv

    return ShmemSubprocVecEnv(factories, start_method="spawn")


def train_ppo(
    config: AppConfig,
    callbacks: Sequence[BaseCallback] | None = None,
    *,
    force_dummy_vec_env: bool = False,
    run_dir: Path | None = None,
    enable_trajectory_viz: bool = True,
) -> Path:
    """Train PPO agent on the balloon environment.

    Returns:
        Каталог текущего run-а (``PPO_N`` с подкаталогами ``tb`` и ``trajectories``).
    """
    from diplom.train.trajectory_callback import TrajectoryVisualizationCallback

    logdir = config.training.logdir
    logdir.mkdir(parents=True, exist_ok=True)
    device = resolve_torch_device(config.training.device)
    ppo_verbose = config.training.verbose
    print(f"[train_ppo] Using device={device}")  # noqa: T201

    if run_dir is None:
        run_dir = next_run_dir(logdir)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = run_dir / "trajectories"
    print(f"[train_ppo] Run directory: {run_dir}")  # noqa: T201

    trajectory_steps_dir = traj_dir if enable_trajectory_viz else None
    vec_env = _make_vec_env(
        config,
        force_dummy=force_dummy_vec_env,
        trajectory_steps_dir=trajectory_steps_dir,
    )
    if config.training.use_worker_policy_rollout and not force_dummy_vec_env and config.training.n_envs > 1:
        print("[train_ppo] VecEnv: PolicyShmemSubprocVecEnv (гибрид worker+shmem)")  # noqa: T201
    elif not force_dummy_vec_env and config.training.n_envs > 1:
        print("[train_ppo] VecEnv: ShmemSubprocVecEnv (policy в main, --main-policy-rollout)")  # noqa: T201
    info_callback = InfoLoggingCallback()

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
        TrajectoryVisualizationCallback(output_dir=traj_dir) if enable_trajectory_viz else None
    )

    model_path = logdir / "ppo_model.zip"
    use_worker_rollout = (
        config.training.use_worker_policy_rollout
        and not force_dummy_vec_env
        and config.training.n_envs > 1
    )
    ppo_cls: type[PPO] = PPO
    if use_worker_rollout:
        from diplom.train.ppo_worker_rollout import WorkerRolloutPPO

        ppo_cls = WorkerRolloutPPO

    model: PPO

    def _save_model() -> None:
        model.save(logdir / "ppo_model")
        print(f"[train_ppo] Модель сохранена в {model_path}")  # noqa: T201

    try:
        # Если в logdir уже есть сохранённая модель, продолжаем обучение с неё.
        # Иначе инициализируем новую.
        if model_path.exists():
            print(f"[train_ppo] Продолжаем обучение из {model_path}")  # noqa: T201
            model = ppo_cls.load(model_path, env=vec_env, device=device)
            model.tensorboard_log = str(run_dir)
        else:
            model = ppo_cls(
                policy="MlpPolicy",
                env=vec_env,
                # Собираем больше опыта перед каждым обновлением — лучше
                # использование данных и меньше накладных расходов обновления.
                n_steps=config.training.ppo_n_steps,
                # 512 делит rollout_size=4096*n_envs нацело и эффективнее на MPS/CPU.
                batch_size=512,
                n_epochs=10,
                learning_rate=3e-4,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                # Небольшой энтропийный бонус удерживает агента от преждевременной
                # конвергенции к детерминированной политике.
                ent_coef=0.005,
                vf_coef=0.5,
                max_grad_norm=0.5,
                verbose=ppo_verbose,
                tensorboard_log=str(run_dir),
                seed=config.training.seed,
                device=device,
            )
        model.verbose = ppo_verbose
        extra_callbacks = list(callbacks) if callbacks is not None else []
        learn_callbacks: list[BaseCallback] = [info_callback]
        if traj_callback is not None:
            learn_callbacks.append(traj_callback)
        learn_callbacks.extend(extra_callbacks)
        try:
            model.learn(
                total_timesteps=config.training.total_timesteps,
                callback=CallbackList(learn_callbacks),
                reset_num_timesteps=True,
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
