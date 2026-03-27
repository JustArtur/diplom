"""CLI-интерфейс для симулятора стратостата и RL-обучения."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from datetime import datetime

import typer
from dotenv import load_dotenv

from diplom.data.era5_download import DEFAULT_PRESSURE_LEVELS, DEFAULT_VARIABLES

# Загружаем переменные окружения (.env) — ключи CDS API и т.п.
load_dotenv()

# Главный объект Typer CLI
app = typer.Typer(help="CLI утилиты для симулятора стратостата и RL.")


# ──────────────────── download ────────────────────

@app.command()
def download(
    outfile: Path = typer.Option(Path("data/raw/era5_sample.nc"), "--outfile", "-o"),
    north: float = typer.Option(57.0, help="Северная граница широты"),
    west: float = typer.Option(56.0, help="Западная граница долготы"),
    south: float = typer.Option(52.0, help="Южная граница широты"),
    east: float = typer.Option(58.0, help="Восточная граница долготы"),
    start: str = typer.Option("2024-07-01", help="Начало периода YYYY-MM-DD"),
    end: str = typer.Option("2024-07-02", help="Конец периода YYYY-MM-DD"),
    level: list[str] = typer.Option(
        list(DEFAULT_PRESSURE_LEVELS),
        "--level", "-l",
        help="Уровни давления hPa; можно повторять.",
        show_default=False,
    ),
    variable: list[str] = typer.Option(
        list(DEFAULT_VARIABLES),
        "--var", "-v",
        help="Имена переменных CDS; можно повторять.",
        show_default=False,
    ),
) -> None:
    """Скачать подмножество ERA5 в NetCDF."""
    from diplom.data.era5_download import download_era5_pressure

    download_era5_pressure(
        outfile=outfile,
        north=north, west=west, south=south, east=east,
        start=start, end=end,
        pressure_levels=level,
        variables=variable,
    )



# ──────────────────── viz_real ────────────────────

@app.command()
def viz_real(
    data: Path = typer.Option(Path("data/raw/era5_sample.nc"), "--data", help="Путь к NetCDF ERA5"),
    origin_lat: float = typer.Option(54.5, "--lat",help="Широта точки отсчёта"),
    origin_lon: float = typer.Option(57.0, "--lon", help="Долгота точки отсчёта"),
    start_time: datetime = typer.Option("2024-07-01", help="Время слоя симуляции"),
) -> None:
    """Запуск PyVista-визуализации на реальном ветре."""

    from diplom.viz.visualization_runner import VisualizationRunner

    VisualizationRunner().run_real(
        data_path=data,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        start_time=start_time,
    )


# ──────────────────── Точка входа ────────────────────

def main() -> None:
    """Точка входа CLI."""
    app()


if __name__ == "__main__":
    main()
