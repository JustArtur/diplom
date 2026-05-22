"""PPO-модель ``lstm`` — RecurrentPPO с MlpLstmPolicy (sb3-contrib).

CLI: ``--model lstm``

Архитектура
-----------
- Policy: ``MlpLstmPolicy`` (RecurrentPPO из sb3-contrib).
- Перед MLP: LSTM encoder (hidden=256, 1 слой).
- Actor head: [128] → action mean.
- Critic head: [128] → value (отдельный LSTM: ``enable_critic_lstm=True``).
- ``shared_lstm=False`` — actor и critic не делят одну LSTM.

Гиперпараметры
--------------
- ``LSTM_HIDDEN_SIZE = 256``
- ``N_LSTM_LAYERS = 1``
- ``NET_ARCH = {"pi": [128], "vf": [128]}`` — после LSTM.
- log_std: те же границы, что у ``default``.

Зачем LSTM
----------
Задача зависит от истории (ветер во времени, прошлые решения по Z).
LSTM получает последовательность obs и может запоминать контекст,
когда Markov obs недостаточен.

Особенности обучения
--------------------
- Runner использует RecurrentPPO вместо PPO.
- SubprocVecEnv без worker-rollout shmem (другой путь collect_rollouts).
- Эпизодные hidden state сбрасываются на done.

Совместимость
--------------
- obs: рекомендуется ``default`` (33) с temporal/probe фичами.
- Чекпоинт ``default`` ↔ ``lstm`` **не взаимозаменяемы**.

Экспорт: ``SPEC`` с ``recurrent=True``.
"""

from __future__ import annotations

from diplom.rl.ppo.models.default import ModelSpec

POLICY_TYPE = "MlpLstmPolicy"
NET_ARCH: dict[str, list[int]] = {"pi": [128], "vf": [128]}
LSTM_HIDDEN_SIZE = 256
N_LSTM_LAYERS = 1
ENABLE_CRITIC_LSTM = True
SHARED_LSTM = False
LOG_STD_INIT = -1.5
LOG_STD_MIN = -1.609
LOG_STD_MAX = -0.693

SPEC = ModelSpec(
    name="lstm",
    policy_type=POLICY_TYPE,
    net_arch=NET_ARCH,
    log_std_init=LOG_STD_INIT,
    log_std_min=LOG_STD_MIN,
    log_std_max=LOG_STD_MAX,
    recurrent=True,
    lstm_hidden_size=LSTM_HIDDEN_SIZE,
    n_lstm_layers=N_LSTM_LAYERS,
    enable_critic_lstm=ENABLE_CRITIC_LSTM,
    shared_lstm=SHARED_LSTM,
)
