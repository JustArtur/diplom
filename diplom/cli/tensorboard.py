from __future__ import annotations

from pathlib import Path

import typer


def export_tensorboard(
    path: Path = typer.Argument(
        ...,
        help="Файл events.out.tfevents.* или каталог (tb_1, {датасет}/PPO_N, ppo)",
    ),
    recursive: bool = typer.Option(
        True,
        "--recursive/--no-recursive",
        help="Искать event-файлы в подкаталогах",
    ),
    summary: bool = typer.Option(
        True,
        "--summary/--no-summary",
        help="Сводка ключевых метрик в .scalars.summary.txt/.json",
    ),
) -> None:
    """Экспорт scalar-метрик TensorBoard в CSV рядом с каждым event-файлом.

    Создаёт файлы вида ``events.out.tfevents.<id>.scalars.csv`` с колонками:
    tag, step, value, wall_time и сводку ``*.scalars.summary.txt`` / ``*.scalars.summary.json``.

    \b
    Примеры:

      diplom export-tensorboard ppo/{датасет}/PPO_25/tb_1

      diplom export-tensorboard ppo/{датасет}/PPO_25/tb_1/events.out.tfevents.1779218441.host.0
    """
    from diplom.dev.tensorboard.export import export_tensorboard_path

    try:
        results = export_tensorboard_path(path, recursive=recursive, write_summary=summary)
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    for item in results:
        typer.echo(f"{item.output}  ({item.rows} строк, из {item.source.name})")
        if item.summary_txt is not None:
            typer.echo(f"{item.summary_txt}")
        if item.summary_json is not None:
            typer.echo(f"{item.summary_json}")
