# Реестр PPO-моделей (default, explore, lstm). Выбор: diplom train-ppo --model <name>.
#
# Размер входа сети должен совпадать с OBS_DIM выбранной obs-модели.

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
