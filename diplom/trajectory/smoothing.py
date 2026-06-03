from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

SmoothMethod = Literal["transition", "ema", "moving_average"]


@dataclass(frozen=True, slots=True)
class SmoothingStats:
    transition_count: int
    max_action_delta: float
    max_ramp_steps: int
    segment_steps: int


def _clip_actions(actions: list[float], action_limit: float) -> list[float]:
    limit = float(action_limit)
    return [float(np.clip(value, -limit, limit)) for value in actions]


def _cosine_ease(t: float) -> float:
    # S-кривая без резких производных на концах
    x = float(np.clip(t, 0.0, 1.0))
    return 0.5 - 0.5 * float(np.cos(np.pi * x))


def _extract_action_plateaus(actions: list[float]) -> list[tuple[int, int, float]]:
    # Индексы [start, end) и значение для каждого платo постоянного action
    if not actions:
        return []

    plateaus: list[tuple[int, int, float]] = []
    start = 0
    value = float(actions[0])
    for idx in range(1, len(actions)):
        current = float(actions[idx])
        if current != value:
            plateaus.append((start, idx, value))
            start = idx
            value = current
    plateaus.append((start, len(actions), value))
    return plateaus


def count_action_transitions(actions: list[float]) -> SmoothingStats:
    plateaus = _extract_action_plateaus(actions)
    if len(plateaus) <= 1:
        return SmoothingStats(
            transition_count=0,
            max_action_delta=0.0,
            max_ramp_steps=0,
            segment_steps=len(actions),
        )

    max_delta = 0.0
    for left, right in zip(plateaus, plateaus[1:], strict=False):
        max_delta = max(max_delta, abs(right[2] - left[2]))
    return SmoothingStats(
        transition_count=len(plateaus) - 1,
        max_action_delta=max_delta,
        max_ramp_steps=0,
        segment_steps=len(actions),
    )


def _ramp_length(plateau_len: int, blend_fraction: float) -> int:
    if plateau_len <= 0:
        return 0
    if plateau_len == 1:
        return 1
    fraction = float(np.clip(blend_fraction, 0.05, 1.0))
    return max(1, min(plateau_len, int(round(fraction * plateau_len))))


def smooth_actions_keyframe(
    actions: list[float],
    *,
    blend_fraction: float = 0.6,
    action_limit: float | None = None,
) -> tuple[list[float], SmoothingStats]:
    # Сгладить участок action с последней точки s до текущего шага
    # На каждом скачке, S-кривая через blend_fraction (по умолчанию 0.6)
    # предыдущего и следующего платo вокруг границы
    count = len(actions)
    if count <= 1:
        return [float(value) for value in actions], SmoothingStats(0, 0.0, 0, count)

    plateaus = _extract_action_plateaus(actions)
    if len(plateaus) <= 1:
        return [float(value) for value in actions], SmoothingStats(0, 0.0, 0, count)

    result = [float(value) for value in actions]
    max_ramp_span = 0
    transition_count = 0
    max_delta = 0.0

    for plateau_idx in range(1, len(plateaus)):
        prev_start, prev_end, prev_value = plateaus[plateau_idx - 1]
        curr_start, curr_end, curr_value = plateaus[plateau_idx]
        if prev_value == curr_value:
            continue

        transition_count += 1
        max_delta = max(max_delta, abs(curr_value - prev_value))

        prev_len = prev_end - prev_start
        curr_len = curr_end - curr_start
        prev_ramp = _ramp_length(prev_len, blend_fraction)
        curr_ramp = _ramp_length(curr_len, blend_fraction)

        left = max(prev_start, curr_start - prev_ramp)
        right = min(count, curr_start + curr_ramp)
        if right <= left:
            right = min(count, left + 2)

        span = max(1, right - left - 1)
        max_ramp_span = max(max_ramp_span, right - left)

        for step_idx in range(left, right):
            t = _cosine_ease((step_idx - left) / span)
            result[step_idx] = prev_value + (curr_value - prev_value) * t

    stats = SmoothingStats(
        transition_count=transition_count,
        max_action_delta=max_delta,
        max_ramp_steps=max_ramp_span,
        segment_steps=count,
    )
    if action_limit is not None:
        return _clip_actions(result, action_limit), stats
    return result, stats


def smooth_actions_transition(
    actions: list[float],
    *,
    blend_fraction: float = 0.6,
    action_limit: float | None = None,
) -> list[float]:
    # Keyframe-рампы (основной режим для manual s)
    smoothed, _stats = smooth_actions_keyframe(
        actions,
        blend_fraction=blend_fraction,
        action_limit=action_limit,
    )
    return smoothed


def smooth_actions_ema(
    actions: list[float],
    *,
    alpha: float = 0.25,
    action_limit: float | None = None,
) -> list[float]:
    if not actions:
        return []

    weight = float(np.clip(alpha, 0.01, 1.0))
    smoothed = [float(actions[0])]
    for value in actions[1:]:
        prev = smoothed[-1]
        smoothed.append(weight * float(value) + (1.0 - weight) * prev)

    if action_limit is not None:
        return _clip_actions(smoothed, action_limit)
    return smoothed


def smooth_actions_moving_average(
    actions: list[float],
    *,
    window: int = 8,
    action_limit: float | None = None,
) -> list[float]:
    if not actions:
        return []

    size = max(1, int(window))
    arr = np.asarray(actions, dtype=np.float64)
    kernel = np.ones(size, dtype=np.float64) / size
    padded = np.pad(arr, (size // 2, size - 1 - size // 2), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    result = [float(value) for value in smoothed[: len(actions)]]

    if action_limit is not None:
        return _clip_actions(result, action_limit)
    return result


def smooth_actions(
    actions: list[float],
    *,
    method: SmoothMethod = "transition",
    blend_fraction: float = 0.6,
    alpha: float = 0.25,
    window: int = 8,
    action_limit: float | None = None,
) -> tuple[list[float], SmoothingStats]:
    # Сгладить последовательность action перед replay в физике
    if method == "ema":
        smoothed = smooth_actions_ema(actions, alpha=alpha, action_limit=action_limit)
        return smoothed, count_action_transitions(actions)
    if method == "moving_average":
        smoothed = smooth_actions_moving_average(actions, window=window, action_limit=action_limit)
        return smoothed, count_action_transitions(actions)
    return smooth_actions_keyframe(
        actions,
        blend_fraction=blend_fraction,
        action_limit=action_limit,
    )
