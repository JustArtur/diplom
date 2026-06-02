from __future__ import annotations

from pathlib import Path

import typer

from diplom.trajectory.demos import export_demo_dataset


def export_demonstrations(
    source: Path = typer.Option(
        Path("runs/manual"),
        "--source",
        "-s",
        help="Каталог run-а или trajectories/ с успешными эпизодами",
    ),
    output: Path = typer.Option(
        Path("runs/demos/demo_dataset.npz"),
        "--output",
        "-o",
        help="Куда сохранить NPZ-датасет демонстраций",
    ),
    max_episodes: int | None = typer.Option(
        None,
        "--max-episodes",
        min=1,
        help="Ограничить число эпизодов, попавших в экспорт",
    ),
) -> None:
    """Экспортировать успешные JSONL-эпизоды в компактный npz для BC/pretraining."""
    try:
        summary = export_demo_dataset(source, output, max_episodes=max_episodes)
    except ValueError as exc:
        typer.echo(f"[ошибка] {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"source={summary.source_dir}")
    typer.echo(f"output={summary.output_path}")
    typer.echo(f"summary={summary.summary_path}")
    typer.echo(
        f"episodes={summary.episode_count} transitions={summary.transition_count} "
        f"obs_dim={summary.obs_dim} action_dim={summary.action_dim}"
    )
