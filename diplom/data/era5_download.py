from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import List

import typer

from diplom.config import DownloadConfig


def download_era5_pressure(config: DownloadConfig) -> None:
    # cdsapi лучше импортировать лениво: модуль скачивания не должен тянуть зависимость на импорт.
    import cdsapi  # lazy import to avoid dependency when unused

    # Проверяем доступность учётных данных и создаём каталог назначения заранее.
    _check_credentials()
    _ensure_parent(config.outfile)

    client = cdsapi.Client()
    # CDS ждёт развернутый список дат, поэтому сначала нормализуем диапазон в календарные дни.
    days = _date_range(config.start, config.end)
    hours = [f"{h:02d}:00" for h in range(24)]

    # Сборка словаря запроса к ERA5 pressure-levels.
    request = {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": list(config.variables),
        "pressure_level": list(config.pressure_levels),
        "year": sorted({d[:4] for d in days}),
        "month": sorted({d[5:7] for d in days}),
        "day": sorted({d[8:10] for d in days}),
        "time": hours,
        "area": [config.north, config.west, config.south, config.east],
    }

    typer.secho(
        f"Requesting ERA5 pressure-levels for {len(days)} day(s) to {config.outfile}...",
        fg=typer.colors.CYAN,
    )
    client.retrieve("reanalysis-era5-pressure-levels", request, str(config.outfile))
    typer.secho("Done.", fg=typer.colors.GREEN)


def _date_range(start: str, end: str) -> List[str]:
    start_day = dt.date.fromisoformat(start)
    end_day = dt.date.fromisoformat(end)

    if end_day < start_day:
        raise typer.BadParameter("end date must be >= start date")

    days = []
    cur = start_day

    # Идём по календарным дням включительно, чтобы CDS получил полный интервал.
    while cur <= end_day:
        days.append(cur.isoformat())
        cur += dt.timedelta(days=1)

    return days


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _check_credentials() -> None:
    env_key = os.environ.get("CDSAPI_KEY")
    env_url = os.environ.get("CDSAPI_URL")

    if env_key and env_url:
        return

    typer.secho(
        "Ошибка: отсутствуют учётные данные CDS API. "
        "Установите переменные окружения CDSAPI_URL и CDSAPI_KEY.",
        fg=typer.colors.RED,
        err=True,
    )
    raise typer.Exit(code=1)
