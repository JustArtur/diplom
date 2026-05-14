from __future__ import annotations

import json
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Sequence, Union

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from diplom.config import AppConfig, EnvironmentConfig, WindConfig
from diplom.envs.balloon_env import BalloonEnv
from diplom.envs.factory import build_env
from diplom.train.info_logging_callback import InfoLoggingCallback


def _env_factory(env_config: EnvironmentConfig, wind_config: WindConfig) -> BalloonEnv:
    """Создаёт одну среду внутри рабочего подпроцесса.

    Функция должна быть определена на уровне модуля (не lambda и не closure),
    чтобы pickle мог сериализовать её при передаче в SubprocVecEnv.
    """
    return build_env(env_config, wind_config)


def _make_vec_env(config: AppConfig) -> Union[DummyVecEnv, SubprocVecEnv]:
    n_envs = max(1, config.training.n_envs)
    # Включаем рандомизацию старта — нужна при обучении.
    env_config = replace(
        config.environment,
        randomize_start_state=True,
        randomize_start_time=True,
    )
    # partial — picklable, в отличие от lambda; именно это позволяет SubprocVecEnv
    # передавать фабрику через pickle в дочерний процесс.
    factory = partial(_env_factory, env_config=env_config, wind_config=config.wind)

    if n_envs == 1:
        # Один процесс — нет смысла в оверхеде IPC.
        return DummyVecEnv([factory])

    # start_method="spawn" безопасен на macOS и Windows (fork может дедлочить
    # внутренние потоки xarray/netCDF4).
    return SubprocVecEnv([factory] * n_envs, start_method="spawn")


def _select_device() -> str:
    """Выбрать устройство для обучения PPO.

    Для MLP-политик Stable-Baselines3 обычно быстрее и стабильнее работает на CPU,
    чем на GPU/MPS, поэтому по умолчанию используем CPU.
    """
    return "cpu"



def _next_run_dir(trajectories_root: Path) -> Path:
    """Вернуть следующую по счёту директорию ppo_NNN внутри trajectories_root.

    Просматривает существующие папки вида ppo_000, ppo_001, … и возвращает
    ppo_{max+1:03d}. Если папок нет — возвращает ppo_000.
    """
    existing = sorted(
        p for p in trajectories_root.glob("ppo_*") if p.is_dir()
    )
    if not existing:
        return trajectories_root / "ppo_000"
    last_num = int(existing[-1].name.split("_")[-1])
    return trajectories_root / f"ppo_{last_num + 1:03d}"


def train_ppo(config: AppConfig, callbacks: Sequence[BaseCallback] | None = None) -> None:
    """Train PPO agent on the balloon environment."""
    from diplom.train.trajectory_callback import TrajectoryVisualizationCallback

    logdir = config.training.logdir
    logdir.mkdir(parents=True, exist_ok=True)
    device = _select_device()
    print(f"[train_ppo] Using device={device}")  # noqa: T201

    vec_env = _make_vec_env(config)
    eval_env = None
    info_callback = InfoLoggingCallback()

    traj_dir = _next_run_dir(logdir / "trajectories")
    traj_callback = TrajectoryVisualizationCallback(
        output_dir=traj_dir,
        verbose=1,
    )

    model_path = logdir / "ppo_model.zip"
    model: PPO
    def _save_model() -> None:
        model.save(logdir / "ppo_model")
        print(f"[train_ppo] Модель сохранена в {model_path}")  # noqa: T201

    try:
        # Если в logdir уже есть сохранённая модель, продолжаем обучение с неё.
        # Иначе инициализируем новую.
        if model_path.exists():
            print(f"[train_ppo] Продолжаем обучение из {model_path}")  # noqa: T201
            model = PPO.load(model_path, env=vec_env, device=device)
        else:
            model = PPO(
                policy="MlpPolicy",
                env=vec_env,
                # Собираем больше опыта перед каждым обновлением — лучше
                # использование данных и меньше накладных расходов обновления.
                n_steps=4096,
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
                verbose=1,
                tensorboard_log=str(logdir / "tb"),
                seed=config.training.seed,
                device=device,
            )
        extra_callbacks = list(callbacks) if callbacks is not None else []
        try:
            model.learn(
                total_timesteps=config.training.total_timesteps,
                callback=CallbackList([info_callback, traj_callback, *extra_callbacks]),
                reset_num_timesteps=False,
            )
        except KeyboardInterrupt:
            _save_model()
            raise
        else:
            _save_model()

        # Простой контрольный прогон на отдельной среде после обучения.
        # Здесь отключаем рандомизацию, чтобы получить детерминированную оценку.
        eval_env = build_env(env_config=config.environment, wind_config=config.wind)
        obs, _ = eval_env.reset(seed=config.training.seed + 1)
        done = False
        truncated = False
        ep_reward = 0.0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _info = eval_env.step(action)
            ep_reward += float(reward)

        (logdir / "eval.json").write_text(json.dumps({"episode_reward": ep_reward}, indent=2, ensure_ascii=False))
    finally:
        if eval_env is not None:
            eval_env.close()
        vec_env.close()
