from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch as th
from stable_baselines3 import PPO
from torch.utils.data import DataLoader, TensorDataset

from diplom.trajectory.demos import DemoDataset, load_demo_dataset


@dataclass(frozen=True, slots=True)
class DemoPretrainingSummary:
    dataset_path: Path
    sample_count: int
    epochs: int
    batch_size: int
    average_loss: float


def _ensure_action_shape(actions: np.ndarray) -> np.ndarray:
    if actions.ndim == 1:
        return actions[:, None]
    return actions


def _validate_demo_dataset(model: PPO, dataset: DemoDataset) -> None:
    obs_shape = getattr(model.observation_space, "shape", None)
    action_shape = getattr(model.action_space, "shape", None)
    if obs_shape is None or action_shape is None:
        raise ValueError("Модель PPO должна работать с Box observation/action spaces")

    if dataset.sample_count == 0:
        raise ValueError("Датасет демонстраций пуст")

    if dataset.obs_dim != int(obs_shape[0]):
        raise ValueError(
            f"Размерность observation в датасете ({dataset.obs_dim}) не совпадает с моделью ({obs_shape[0]})"
        )
    if dataset.action_dim != int(action_shape[0]):
        raise ValueError(
            f"Размерность action в датасете ({dataset.action_dim}) не совпадает с моделью ({action_shape[0]})"
        )


def pretrain_policy_on_demo_dataset(
    model: PPO,
    dataset_path: Path,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float | None = None,
    max_grad_norm: float | None = None,
) -> DemoPretrainingSummary:
    if epochs <= 0:
        raise ValueError("epochs должен быть > 0")
    if batch_size <= 0:
        raise ValueError("batch_size должен быть > 0")

    if getattr(model.policy, "lstm_actor", None) is not None or getattr(model.policy, "shared_lstm", None) is not None:
        raise ValueError(
            "Demo pretraining пока поддерживает только feed-forward PPO-политики. "
            "Для recurrent-модели сначала используйте `--model default` или `--model explore`."
        )

    dataset = load_demo_dataset(dataset_path)
    _validate_demo_dataset(model, dataset)

    observations = th.as_tensor(dataset.observations, dtype=th.float32, device=model.device)
    actions = th.as_tensor(_ensure_action_shape(dataset.actions), dtype=th.float32, device=model.device)
    loader = DataLoader(
        TensorDataset(observations, actions),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    policy = model.policy
    policy.set_training_mode(True)
    optimizer = policy.optimizer
    original_lrs = [group["lr"] for group in optimizer.param_groups]
    if learning_rate is not None:
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

    total_loss = 0.0
    total_batches = 0
    try:
        for _epoch in range(epochs):
            for batch_obs, batch_actions in loader:
                _values, log_prob, _entropy = policy.evaluate_actions(batch_obs, batch_actions)
                loss = -log_prob.mean()
                optimizer.zero_grad()
                loss.backward()
                if max_grad_norm is not None:
                    th.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()
                total_loss += float(loss.item())
                total_batches += 1
    finally:
        if learning_rate is not None:
            for group, original_lr in zip(optimizer.param_groups, original_lrs, strict=False):
                group["lr"] = original_lr
        policy.set_training_mode(False)

    average_loss = total_loss / max(total_batches, 1)
    return DemoPretrainingSummary(
        dataset_path=Path(dataset_path).resolve(),
        sample_count=dataset.sample_count,
        epochs=epochs,
        batch_size=batch_size,
        average_loss=average_loss,
    )
