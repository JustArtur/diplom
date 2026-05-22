from __future__ import annotations

from pathlib import Path

DEFAULT_ERA5_DATA_DIR = Path("data")
ERA5_PREVIEW_DATA_DIR = DEFAULT_ERA5_DATA_DIR / "preview"
ERA5_TRAINING_DATA_DIR = DEFAULT_ERA5_DATA_DIR / "training"
ERA5_TRAINING_MANIFEST_PATH = ERA5_TRAINING_DATA_DIR / "datasets_manifest.toml"
ERA5_CACHE_DIR = DEFAULT_ERA5_DATA_DIR / "cache"
ERA5_WIND_CACHE_DIR = ERA5_CACHE_DIR / "wind"
ERA5_DOWNLOAD_CHUNKS_DIR = ERA5_CACHE_DIR / "chunks"

DEFAULT_ERA5_NORTH = 35.0
DEFAULT_ERA5_SOUTH = -5.0
DEFAULT_ERA5_WEST = 50.0
DEFAULT_ERA5_EAST = 110.0
DEFAULT_ERA5_START = "2024-10-05"
DEFAULT_ERA5_END = "2024-10-19"


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
    directory: Path = ERA5_TRAINING_DATA_DIR,
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


def list_era5_datasets(directory: Path = ERA5_TRAINING_DATA_DIR) -> list[Path]:
    """Все NetCDF-датасеты ERA5 в каталоге (отсортированы по имени файла)."""
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.nc"))


def resolve_era5_dataset_path(
    name: str | Path,
    *,
    data_dir: Path = ERA5_TRAINING_DATA_DIR,
) -> Path:
    """Разрешить имя или путь датасета в существующий NetCDF-файл."""
    path = Path(name)

    if path.is_file():
        return path

    candidates: list[Path] = []
    if path.suffix == ".nc":
        candidates.append(data_dir / path.name)
    else:
        candidates.extend((data_dir / f"{path.name}.nc", data_dir / path.name))

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    available = list_era5_datasets(data_dir)
    hint = ", ".join(p.name for p in available) if available else "(каталог пуст)"
    raise ValueError(f"Датасет «{name}» не найден. Доступные в {data_dir}: {hint}")


def era5_dataset_title(path: Path) -> str:
    """Человекочитаемое имя датасета (без расширения .nc) для заголовка графика."""
    return path.stem


def training_logdir_for_dataset(
    dataset_path: Path,
    parent_logdir: Path,
    *,
    experiment_name: str | None = None,
) -> Path:
    """Каталог артефактов: ``{parent}/{experiment_name|dataset_stem}``."""
    dir_name = (
        experiment_name.strip()
        if experiment_name and experiment_name.strip()
        else dataset_path.stem
    )
    return parent_logdir / dir_name


def resolve_dataset_reference(
    reference: str | int,
    *,
    data_dir: Path = ERA5_TRAINING_DATA_DIR,
    manifest_path: Path = ERA5_TRAINING_MANIFEST_PATH,
) -> Path:
    """Разрешить имя, путь или числовой id (#1 из datasets_manifest.toml) в NetCDF."""
    if isinstance(reference, int) or str(reference).isdigit():
        dataset_id = int(reference)
        if not manifest_path.is_file():
            raise ValueError(
                f"Для --dataset {dataset_id} нужен манифест {manifest_path}"
            )
        from diplom.data.era5_manifest import load_training_manifest

        manifest = load_training_manifest(manifest_path)
        for spec in manifest.datasets:
            if spec.id == dataset_id:
                return resolve_era5_dataset_path(spec.stem, data_dir=data_dir)
        ids = ", ".join(str(spec.id) for spec in manifest.datasets)
        raise ValueError(f"Датасет id={dataset_id} не найден в манифесте. Доступные id: {ids}")
    return resolve_era5_dataset_path(str(reference), data_dir=data_dir)


def wind_plot_html_path(dataset_path: Path, output_dir: Path) -> Path:
    """Путь к HTML-графику ветра для датасета: {output_dir}/{stem}.html."""
    return output_dir / f"{dataset_path.stem}.html"


def wind_cache_value_path(source_path: Path) -> Path:
    """Путь к memmap-кэшу интерполятора ветра для NetCDF-датасета."""
    return ERA5_WIND_CACHE_DIR / f"{source_path.name}.wind-cache.npy"


def wind_cache_meta_path(source_path: Path) -> Path:
    """Путь к JSON-метаданным кэша интерполятора ветра."""
    return ERA5_WIND_CACHE_DIR / f"{source_path.name}.wind-cache.json"


def download_chunks_dir(outfile: Path) -> Path:
    """Каталог для промежуточных NetCDF-чанков при скачивании ERA5."""
    return ERA5_DOWNLOAD_CHUNKS_DIR / f"{outfile.stem}.chunks"
