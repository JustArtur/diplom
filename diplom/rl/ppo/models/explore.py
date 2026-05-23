"""PPO-модель ``explore`` — более широкая и более стохастическая политика.

CLI: ``--model explore``

Архитектура
-----------
- Policy: ``MlpPolicy`` (Stable-Baselines3 PPO).
- Actor (pi): Linear → [256] → ReLU → [256] → ReLU → [128] → ReLU → action mean.
- Critic (vf): Linear → [256] → ReLU → [256] → ReLU → [128] → ReLU → value.
- Action: 1D continuous (кг/с накачки), Gaussian с более широким начальным
  и допустимым диапазоном ``log_std``.

Зачем
-----
Модель рассчитана на более активный exploration: она чаще пробует разные
режимы управления, что полезно в задачах со множеством локальных минимумов.
"""

from __future__ import annotations

from diplom.rl.ppo.models.default import ModelSpec

POLICY_TYPE = "MlpPolicy"
NET_ARCH: dict[str, list[int]] = {"pi": [256, 256, 128], "vf": [256, 256, 128]}
LOG_STD_INIT = -0.9
LOG_STD_MIN = -2.0
LOG_STD_MAX = 0.0


SPEC = ModelSpec(
    name="explore",
    policy_type=POLICY_TYPE,
    net_arch=NET_ARCH,
    log_std_init=LOG_STD_INIT,
    log_std_min=LOG_STD_MIN,
    log_std_max=LOG_STD_MAX,
)
