"""Выбор устройства вычислений PyTorch (cpu / cuda / mps)."""

from __future__ import annotations


def resolve_torch_device(kind: str) -> str:
    """Превращает строку запроса в валидное имя устройства для SB3/torch.

    Raises:
        ValueError: неизвестный тип или запрошенное устройство недоступно.
    """
    import torch

    key = kind.strip().lower()
    if key == "cpu":
        return "cpu"
    if key == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("Запрошено CUDA, но torch.cuda.is_available() == False.")
        return "cuda"
    if key == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("Запрошено MPS, но torch.backends.mps.is_available() == False.")
        return "mps"
    raise ValueError(f"Неизвестное устройство: {kind!r}. Допустимо: cpu, cuda, mps.")
