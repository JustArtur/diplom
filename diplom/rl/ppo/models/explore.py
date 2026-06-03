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
