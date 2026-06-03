# Сводка ключевых метрик из CSV, экспортированного из TensorBoard.

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

SUMMARY_TXT_SUFFIX = ".summary.txt"
SUMMARY_JSON_SUFFIX = ".summary.json"

# Метрики PPO / rollout, берём как есть из TB.
CORE_TAGS: tuple[str, ...] = (
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
    "custom/success_rate",
    "train/approx_kl",
    "train/clip_fraction",
    "train/std",
    "train/explained_variance",
    "train/value_loss",
    "train/policy_gradient_loss",
    "train/entropy_loss",
    "time/fps",
)

# Усредняем по env_0, env_1, … на каждом step.
ENV_AGGREGATE_SUFFIXES: tuple[str, ...] = (
    "distance_to_target",
    "horizontal_distance",
    "progress_reward",
    "reward_boundary_term",
)


@dataclass(frozen=True, slots=True)
class MetricStats:
    tag: str
    n: int
    step_min: int
    step_max: int
    first: float
    last: float
    min: float
    max: float
    early5_mean: float
    late5_mean: float

    @property
    def delta_late_early(self) -> float:
        return self.late5_mean - self.early5_mean


@dataclass(frozen=True, slots=True)
class ScalarsSummary:
    csv_path: Path
    unique_tags: int
    total_rows: int
    max_step: int
    metrics: dict[str, MetricStats] = field(default_factory=dict)
    missing_tags: tuple[str, ...] = ()


def _metric_stats(tag: str, points: list[tuple[int, float]]) -> MetricStats | None:
    if not points:
        return None
    points.sort(key=lambda item: item[0])
    steps = [step for step, _ in points]
    values = [value for _, value in points]
    n = len(values)
    early_n = min(5, n)
    late_n = min(5, n)
    return MetricStats(
        tag=tag,
        n=n,
        step_min=steps[0],
        step_max=steps[-1],
        first=values[0],
        last=values[-1],
        min=min(values),
        max=max(values),
        early5_mean=sum(values[:early_n]) / early_n,
        late5_mean=sum(values[-late_n:]) / late_n,
    )


def _load_series(csv_path: Path) -> dict[str, list[tuple[int, float]]]:
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            series[row["tag"]].append((int(row["step"]), float(row["value"])))
    return series


def _aggregate_env_series(
    series: dict[str, list[tuple[int, float]]],
    suffix: str,
) -> list[tuple[int, float]] | None:
    env_tags = [tag for tag in series if tag.endswith(f"/{suffix}")]
    if not env_tags:
        return None
    by_step: dict[int, list[float]] = defaultdict(list)
    for tag in env_tags:
        for step, value in series[tag]:
            by_step[step].append(value)
    return sorted((step, sum(values) / len(values)) for step, values in by_step.items())


def analyze_scalars_csv(csv_path: Path) -> ScalarsSummary:
    # Построить сводку по *.scalars.csv.
    csv_path = csv_path.resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Не найден CSV: {csv_path}")

    series = _load_series(csv_path)
    total_rows = sum(len(points) for points in series.values())
    max_step = max(step for points in series.values() for step, _ in points) if series else 0

    metrics: dict[str, MetricStats] = {}
    missing: list[str] = []

    for tag in CORE_TAGS:
        stats = _metric_stats(tag, series.get(tag, []))
        if stats is None:
            missing.append(tag)
        else:
            metrics[tag] = stats

    for suffix in ENV_AGGREGATE_SUFFIXES:
        tag = f"avg_env/{suffix}"
        merged = _aggregate_env_series(series, suffix)
        stats = _metric_stats(tag, merged or [])
        if stats is None:
            missing.append(tag)
        else:
            metrics[tag] = stats

    return ScalarsSummary(
        csv_path=csv_path,
        unique_tags=len(series),
        total_rows=total_rows,
        max_step=max_step,
        metrics=metrics,
        missing_tags=tuple(missing),
    )


def _fmt_float(value: float) -> str:
    abs_v = abs(value)
    if abs_v >= 1e4 or (abs_v > 0 and abs_v < 1e-3):
        return f"{value:.4g}"
    return f"{value:.4f}"


def _fmt_metric_line(stats: MetricStats) -> str:
    return (
        f"{stats.tag}: n={stats.n} steps {stats.step_min:,}..{stats.step_max:,} | "
        f"first={_fmt_float(stats.first)} last={_fmt_float(stats.last)} | "
        f"early5={_fmt_float(stats.early5_mean)} late5={_fmt_float(stats.late5_mean)} "
        f"Δ={_fmt_float(stats.delta_late_early)}"
    )


def _success_rate_note(stats: MetricStats) -> str:
    return f"success_rate: max={_fmt_float(stats.max)} (n={stats.n})"


def format_summary_text(summary: ScalarsSummary) -> str:
    lines = [
        f"# TensorBoard summary",
        f"csv: {summary.csv_path}",
        f"rows: {summary.total_rows}  tags: {summary.unique_tags}  max_step: {summary.max_step:,}",
        "",
    ]

    sections: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("rollout", ("rollout/ep_rew_mean", "rollout/ep_len_mean", "custom/success_rate")),
        (
            "train",
            (
                "train/std",
                "train/approx_kl",
                "train/clip_fraction",
                "train/explained_variance",
                "train/value_loss",
                "train/policy_gradient_loss",
                "train/entropy_loss",
            ),
        ),
        ("env (avg)", tuple(f"avg_env/{s}" for s in ENV_AGGREGATE_SUFFIXES)),
        ("time", ("time/fps",)),
    )

    for title, tags in sections:
        lines.append(f"## {title}")
        for tag in tags:
            stats = summary.metrics.get(tag)
            if stats is None:
                lines.append(f"{tag}: —")
                continue
            if tag == "custom/success_rate":
                lines.append(_success_rate_note(stats))
            else:
                lines.append(_fmt_metric_line(stats))
        lines.append("")

    if summary.missing_tags:
        lines.append("## missing")
        lines.extend(f"- {tag}" for tag in summary.missing_tags)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _summary_paths(csv_path: Path) -> tuple[Path, Path]:
    name = csv_path.name
    if name.endswith(".scalars.csv"):
        stem = name[: -len(".scalars.csv")]
        parent = csv_path.parent
        return (
            parent / f"{stem}.scalars{SUMMARY_TXT_SUFFIX}",
            parent / f"{stem}.scalars{SUMMARY_JSON_SUFFIX}",
        )
    return (
        csv_path.with_name(f"{name}{SUMMARY_TXT_SUFFIX}"),
        csv_path.with_name(f"{name}{SUMMARY_JSON_SUFFIX}"),
    )


def write_scalars_summary(csv_path: Path) -> tuple[Path, Path]:
    # Записать .scalars.summary.txt и .scalars.summary.json рядом с CSV.
    summary = analyze_scalars_csv(csv_path)
    txt_path, json_path = _summary_paths(csv_path)

    txt_path.write_text(format_summary_text(summary), encoding="utf-8")

    payload = {
        "csv_path": str(summary.csv_path),
        "unique_tags": summary.unique_tags,
        "total_rows": summary.total_rows,
        "max_step": summary.max_step,
        "missing_tags": list(summary.missing_tags),
        "metrics": {
            tag: {**asdict(stats), "delta_late_early": stats.delta_late_early}
            for tag, stats in summary.metrics.items()
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return txt_path, json_path
