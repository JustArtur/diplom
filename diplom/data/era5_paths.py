from __future__ import annotations

from pathlib import Path

DEFAULT_ERA5_DATA_DIR = Path("data")

DEFAULT_ERA5_NORTH = 30.0
DEFAULT_ERA5_SOUTH = -10.0
DEFAULT_ERA5_WEST = 60.0
DEFAULT_ERA5_EAST = 110.0
DEFAULT_ERA5_START = "2024-05-01"
DEFAULT_ERA5_END = "2024-05-01"


def format_coord_for_filename(value: float) -> str:
    return f"{value:g}"


def format_date_for_filename(value: str) -> str:
    return value.split("T")[0].split(" ")[0]


def era5_outfile_for_bounds(
    *,
    north: float,
    south: float,
    west: float,
    east: float,
    start: str,
    end: str,
    directory: Path = Path("data"),
) -> Path:
    parts = (
        format_coord_for_filename(north),
        format_coord_for_filename(south),
        format_coord_for_filename(west),
        format_coord_for_filename(east),
        format_date_for_filename(start),
        format_date_for_filename(end),
    )
    return directory / f"era5_{'_'.join(parts)}.nc"


DEFAULT_ERA5_OUTFILE = era5_outfile_for_bounds(
    north=DEFAULT_ERA5_NORTH,
    south=DEFAULT_ERA5_SOUTH,
    west=DEFAULT_ERA5_WEST,
    east=DEFAULT_ERA5_EAST,
    start=DEFAULT_ERA5_START,
    end=DEFAULT_ERA5_END,
)


def list_era5_datasets(directory: Path = DEFAULT_ERA5_DATA_DIR) -> list[Path]:
    """Все NetCDF-датасеты ERA5 в каталоге (отсортированы по имени файла)."""
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.nc"))


def era5_dataset_title(path: Path) -> str:
    """Человекочитаемое имя датасета (без расширения .nc) для заголовка графика."""
    return path.stem


def wind_plot_html_path(dataset_path: Path, output_dir: Path) -> Path:
    """Путь к HTML-графику ветра для датасета: {output_dir}/{stem}.html."""
    return output_dir / f"{dataset_path.stem}.html"
