from __future__ import annotations

from pathlib import Path

import typer

from diplom.cli.defaults import DEFAULT_DOWNLOAD_CONFIG
from diplom.config import DownloadConfig
from diplom.data.era5_paths import (
    ERA5_PREVIEW_DATA_DIR,
    ERA5_TRAINING_DATA_DIR,
    ERA5_TRAINING_MANIFEST_PATH,
    era5_outfile_for_bounds,
)


def download(
    outfile: Path | None = typer.Option(
        None,
        "--outfile",
        "-o",
        help="Путь к итоговому NetCDF; по умолчанию data/training/era5_{…}.nc "
        "(с --preview — data/preview/)",
    ),
    north: float = typer.Option(DEFAULT_DOWNLOAD_CONFIG.north, help="Северная граница широты"),
    west: float = typer.Option(DEFAULT_DOWNLOAD_CONFIG.west, help="Западная граница долготы"),
    south: float = typer.Option(DEFAULT_DOWNLOAD_CONFIG.south, help="Южная граница широты"),
    east: float = typer.Option(DEFAULT_DOWNLOAD_CONFIG.east, help="Восточная граница долготы"),
    start: str = typer.Option(DEFAULT_DOWNLOAD_CONFIG.start, help="Начало периода YYYY-MM-DD"),
    end: str = typer.Option(DEFAULT_DOWNLOAD_CONFIG.end, help="Конец периода YYYY-MM-DD"),
    level: list[str] = typer.Option(
        list(DEFAULT_DOWNLOAD_CONFIG.pressure_levels),
        "--level", "-l",
        help="Уровни давления hPa; можно повторять.",
        show_default=False,
    ),
    variable: list[str] = typer.Option(
        list(DEFAULT_DOWNLOAD_CONFIG.variables),
        "--var", "-v",
        help="Имена переменных CDS; можно повторять.",
        show_default=False,
    ),
    chunks_dir: Path | None = typer.Option(
        None,
        "--chunks-dir",
        help="Каталог для чанков NetCDF; по умолчанию data/cache/chunks/{outfile.stem}.chunks",
    ),
    keep_chunks: bool = typer.Option(
        False,
        "--keep-chunks",
        help="Не удалять чанки NetCDF после склейки.",
    ),
    workers: int | None = typer.Option(
        None,
        "--workers", "-j",
        min=1,
        help="Параллельная загрузка чанками по 24 временных точки с последующей склейкой "
        "(2–4 обычно безопасно). Без флага — один запрос CDS сразу в outfile.",
    ),
    hour_step: int = typer.Option(
        DEFAULT_DOWNLOAD_CONFIG.hour_step,
        "--hour-step",
        min=1,
        max=24,
        help="Шаг по часам в запросе CDS: 2 -> 00:00, 02:00, ... 22:00 (12 точек в сутки).",
    ),
    preview: bool = typer.Option(
        False,
        "--preview",
        help="Сохранить в data/preview/ для просмотра ветра (по умолчанию — data/training/).",
    ),
    download_all: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Скачать все датасеты из data/training/datasets_manifest.toml "
        "(пропускать уже существующие; --force — перекачать).",
    ),
    manifest: Path | None = typer.Option(
        None,
        "--manifest",
        help="Путь к TOML-манифесту (по умолчанию data/training/datasets_manifest.toml).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Перекачать, даже если NetCDF уже есть (удаляет файл и чанки).",
    ),
) -> None:
    """Скачать подмножество ERA5 в NetCDF.

    Один регион — укажите --north/--south/--west/--east и --start/--end.

    Все обязательные training-датасеты:

    \b
      diplom download --all --hour-step 8 -j 15
      diplom download --all --force --hour-step 8 -j 15
    """
    from diplom.data.era5_download import download_era5_pressure
    from diplom.data.era5_manifest import download_training_manifest, load_training_manifest

    if download_all:
        if preview:
            typer.echo("[ошибка] --all несовместим с --preview (манифест только для data/training/)", err=True)
            raise typer.Exit(code=1)
        if outfile is not None:
            typer.echo("[ошибка] --all несовместим с --outfile", err=True)
            raise typer.Exit(code=1)

        manifest_path = manifest or ERA5_TRAINING_MANIFEST_PATH
        try:
            training_manifest = load_training_manifest(manifest_path)
        except (FileNotFoundError, ValueError) as exc:
            typer.echo(f"[ошибка] {exc}", err=True)
            raise typer.Exit(code=1) from exc

        download_training_manifest(
            training_manifest,
            pressure_levels=tuple(level),
            variables=tuple(variable),
            hour_step=hour_step,
            workers=workers,
            keep_chunks=keep_chunks,
            chunks_dir=chunks_dir,
            force=force,
        )
        return

    if manifest is not None:
        typer.echo("[ошибка] --manifest используйте вместе с --all", err=True)
        raise typer.Exit(code=1)

    data_dir = ERA5_PREVIEW_DATA_DIR if preview else ERA5_TRAINING_DATA_DIR
    resolved_outfile = outfile or era5_outfile_for_bounds(
        north=north,
        south=south,
        west=west,
        east=east,
        start=start,
        end=end,
        directory=data_dir,
    )

    download_era5_pressure(
        DownloadConfig(
            outfile=resolved_outfile,
            north=north,
            west=west,
            south=south,
            east=east,
            start=start,
            end=end,
            pressure_levels=tuple(level),
            variables=tuple(variable),
            hour_step=hour_step,
        ),
        chunks_dir=chunks_dir,
        keep_chunks=keep_chunks,
        workers=workers,
        force=force,
    )
