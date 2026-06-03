from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.util.tensor_util import make_ndarray

EVENT_FILE_GLOB = "events.out.tfevents.*"
CSV_SUFFIX = ".scalars.csv"


@dataclass(frozen=True, slots=True)
class ExportResult:
    source: Path
    output: Path
    rows: int
    summary_txt: Path | None = None
    summary_json: Path | None = None


def _scalar_from_summary_value(value: object) -> float | None:
    # иногда scalar лежит в tensor, не в simple_value
    if value.HasField("simple_value"):  # type: ignore[attr-defined]
        return float(value.simple_value)  # type: ignore[attr-defined]
    if value.HasField("tensor"):  # type: ignore[attr-defined]
        array = make_ndarray(value.tensor)  # type: ignore[attr-defined]
        if array.size == 0:
            return None
        return float(array.reshape(-1)[0])
    return None


def _iter_scalar_rows(event_path: Path) -> Iterator[tuple[str, int, float, float]]:
    for event in EventFileLoader(str(event_path)).Load():
        if not event.summary or not event.summary.value:
            continue
        step = int(event.step)
        wall_time = float(event.wall_time)
        for value in event.summary.value:
            scalar = _scalar_from_summary_value(value)
            if scalar is None:
                continue
            yield value.tag, step, scalar, wall_time  # type: ignore[attr-defined]


def export_event_file(
    event_path: Path,
    *,
    output_path: Path | None = None,
    write_summary: bool = True,
) -> ExportResult:
    from diplom.dev.tensorboard.summary import write_scalars_summary

    event_path = event_path.resolve()
    if not event_path.is_file():
        raise FileNotFoundError(f"Не найден файл событий TensorBoard: {event_path}")

    out = output_path or Path(f"{event_path}{CSV_SUFFIX}")
    rows = 0
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tag", "step", "value", "wall_time"])
        for row in _iter_scalar_rows(event_path):
            writer.writerow(row)
            rows += 1

    summary_txt: Path | None = None
    summary_json: Path | None = None
    if write_summary and rows > 0:
        summary_txt, summary_json = write_scalars_summary(out)

    return ExportResult(
        source=event_path,
        output=out,
        rows=rows,
        summary_txt=summary_txt,
        summary_json=summary_json,
    )


def find_event_files(root: Path, *, recursive: bool = True) -> list[Path]:
    root = root.resolve()
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"Путь не существует: {root}")

    pattern = f"**/{EVENT_FILE_GLOB}" if recursive else EVENT_FILE_GLOB
    return sorted(
        path for path in root.glob(pattern) if not path.name.endswith(CSV_SUFFIX)
    )


def export_tensorboard_path(
    path: Path,
    *,
    recursive: bool = True,
    write_summary: bool = True,
) -> list[ExportResult]:
    # Экспортирует все events.out.tfevents.* под path (или один файл)
    event_files = find_event_files(path, recursive=recursive)
    if not event_files:
        raise FileNotFoundError(
            f"Нет файлов {EVENT_FILE_GLOB} в {path.resolve()}"
            + (" (попробуйте --recursive)" if not recursive else "")
        )
    return [
        export_event_file(event_path, write_summary=write_summary) for event_path in event_files
    ]
