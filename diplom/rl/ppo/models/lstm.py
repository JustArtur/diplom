# PPO lstm: RecurrentPPO с LSTM.

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
