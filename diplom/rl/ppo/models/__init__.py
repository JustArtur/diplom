"""Реестр PPO-моделей: ``diplom.rl.ppo.models.<name>`` → ``SPEC``.

Каждый модуль экспортирует ``SPEC: ModelSpec`` — конфигурация политики SB3 /
sb3-contrib для ``train_ppo``. Выбор: ``diplom train-ppo --model <name>``.

ModelSpec задаёт
----------------
- ``policy_type`` — класс политики (MlpPolicy / MlpLstmPolicy).
- ``net_arch`` — скрытые слои pi и vf сетей.
- ``log_std_init/min/max`` — диапазон std гауссового action (clip callback).
- ``recurrent`` + LSTM-параметры — для MlpLstmPolicy.

Размер входа политики = ``OBS_DIM`` выбранной obs-модели (``--obs``);
при смене obs нужно новое обучение или совместимая пара obs+checkpoint.

Доступные модели
----------------
default  — MlpPolicy, pi/vf [128, 128], без памяти
lstm     — MlpLstmPolicy, pi/vf [128], LSTM 256, RecurrentPPO
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

from diplom.rl.ppo.models.default import ModelSpec


def list_model_specs() -> list[str]:
    here = Path(__file__).parent
    return sorted(path.stem for path in here.glob("*.py") if path.stem != "__init__")


def get_model_spec(name: str) -> ModelSpec:
    if name not in list_model_specs():
        available = ", ".join(list_model_specs()) or "(пусто)"
        raise ValueError(f"Неизвестная PPO-модель {name!r}. Доступные: {available}")
    return import_module(f"diplom.rl.ppo.models.{name}").SPEC
