from __future__ import annotations

import datetime as dt
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Sequence

import typer

from diplom.config import DownloadConfig

_TIME_DIM_CANDIDATES = ("valid_time", "time")


def download_era5_pressure(
    config: DownloadConfig,
    *,
    chunks_dir: Path | None = None,
    keep_chunks: bool = False,
    workers: int | None = None,
) -> None:
    """Скачать ERA5 в NetCDF.

    Без ``workers`` — один запрос CDS сразу в ``outfile``.
    С ``workers`` — по одному дню на чанк (параллельно при workers > 1), затем склейка.
    """
    _check_credentials()
    _ensure_parent(config.outfile)

    if workers is None:
        _download_single_file(config)
        return

    if workers < 1:
        raise typer.BadParameter("workers must be >= 1")

    days = _date_range(config.start, config.end)
    resolved_chunks_dir = chunks_dir or _default_chunks_dir(config.outfile)
    resolved_chunks_dir.mkdir(parents=True, exist_ok=True)

    hours = _cds_time_hours(config.hour_step)
    chunk_paths = [
        _chunk_path(resolved_chunks_dir, day, hour_step=config.hour_step) for day in days
    ]
    total_days = len(days)

    for index, (day, chunk_path) in enumerate(zip(days, chunk_paths, strict=True), start=1):
        if _is_usable_netcdf(chunk_path):
            typer.secho(
                f"[{index}/{total_days}] Чанк уже есть, пропуск: {chunk_path.name}",
                fg=typer.colors.YELLOW,
            )

    pending = [
        (index, day, chunk_path)
        for index, (day, chunk_path) in enumerate(zip(days, chunk_paths, strict=True), start=1)
        if not _is_usable_netcdf(chunk_path)
    ]

    if pending:
        if workers == 1:
            _download_days_sequential(config, hours=hours, pending=pending, total_days=total_days)
        else:
            typer.secho(
                f"Параллельное скачивание: {len(pending)} дн., workers={workers}",
                fg=typer.colors.CYAN,
            )
            _download_days_parallel(
                config,
                hours=hours,
                pending=pending,
                total_days=total_days,
                workers=workers,
            )

    missing = [path for path in chunk_paths if not _is_usable_netcdf(path)]
    if missing:
        typer.secho(
            "Не все чанки скачаны: "
            + ", ".join(path.name for path in missing),
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    typer.secho(
        f"Склейка {len(chunk_paths)} чанк(ов) в {config.outfile} …",
        fg=typer.colors.CYAN,
    )
    merge_era5_netcdf_files(chunk_paths, config.outfile)

    if keep_chunks:
        typer.secho(f"Чанки сохранены в {resolved_chunks_dir}", fg=typer.colors.YELLOW)
    else:
        _remove_chunks(chunk_paths, resolved_chunks_dir)

    typer.secho("Готово.", fg=typer.colors.GREEN)


def _download_single_file(config: DownloadConfig) -> None:
    """Скачать весь период одним запросом CDS напрямую в outfile."""
    if _is_usable_netcdf(config.outfile):
        typer.secho(
            f"Файл уже есть, пропуск: {config.outfile}",
            fg=typer.colors.YELLOW,
        )
        typer.secho("Готово.", fg=typer.colors.GREEN)
        return

    import cdsapi  # lazy import to avoid dependency when unused

    days = _date_range(config.start, config.end)
    hours = _cds_time_hours(config.hour_step)
    request = _build_request_range(config, days=days, hours=hours)

    period = f"{config.start} … {config.end}" if config.start != config.end else config.start
    typer.secho(
        f"Запрос ERA5 за {period} → {config.outfile.name} …",
        fg=typer.colors.CYAN,
    )
    cdsapi.Client().retrieve(
        "reanalysis-era5-pressure-levels",
        request,
        str(config.outfile),
    )
    typer.secho(f"Готово: {config.outfile}", fg=typer.colors.GREEN)


def merge_era5_netcdf_files(chunk_paths: Sequence[Path], outfile: Path) -> None:
    """Объединить NetCDF-чанки ERA5 по оси времени в один файл."""
    import xarray as xr  # lazy import to avoid dependency when unused

    paths = [Path(path) for path in chunk_paths]
    usable = [path for path in paths if _is_usable_netcdf(path)]
    if not usable:
        raise typer.BadParameter("нет NetCDF-файлов для склейки")

    _ensure_parent(outfile)
    token = uuid.uuid4().hex
    tmp_outfile = outfile.with_name(f".{outfile.name}.{token}.tmp")

    try:
        with xr.open_mfdataset(
            [str(path) for path in usable],
            combine="by_coords",
            parallel=False,
        ) as dataset:
            time_dim = _time_dimension(dataset)
            dataset.sortby(time_dim).to_netcdf(tmp_outfile)
        os.replace(tmp_outfile, outfile)
    except Exception:
        try:
            tmp_outfile.unlink()
        except OSError:
            pass
        raise


def _download_days_sequential(
    config: DownloadConfig,
    *,
    hours: list[str],
    pending: list[tuple[int, str, Path]],
    total_days: int,
) -> None:
    import cdsapi  # lazy import to avoid dependency when unused

    client = cdsapi.Client()
    for index, day, chunk_path in pending:
        request = _build_request(config, day=day, hours=hours)
        typer.secho(
            f"[{index}/{total_days}] Запрос ERA5 за {day} → {chunk_path.name} …",
            fg=typer.colors.CYAN,
        )
        client.retrieve("reanalysis-era5-pressure-levels", request, str(chunk_path))
        typer.secho(
            f"[{index}/{total_days}] Готово: {chunk_path.name}",
            fg=typer.colors.GREEN,
        )


def _download_days_parallel(
    config: DownloadConfig,
    *,
    hours: list[str],
    pending: list[tuple[int, str, Path]],
    total_days: int,
    workers: int,
) -> None:
    log_lock = threading.Lock()
    first_error: list[BaseException] = []

    def _log(message: str, *, fg: int) -> None:
        with log_lock:
            typer.secho(message, fg=fg)

    def _download_one(index: int, day: str, chunk_path: Path) -> None:
        if first_error:
            return

        import cdsapi  # lazy import to avoid dependency when unused

        request = _build_request(config, day=day, hours=hours)
        _log(
            f"[{index}/{total_days}] Запрос ERA5 за {day} → {chunk_path.name} …",
            fg=typer.colors.CYAN,
        )
        try:
            cdsapi.Client().retrieve(
                "reanalysis-era5-pressure-levels",
                request,
                str(chunk_path),
            )
        except BaseException as exc:
            with log_lock:
                if not first_error:
                    first_error.append(exc)
            raise

        _log(
            f"[{index}/{total_days}] Готово: {chunk_path.name}",
            fg=typer.colors.GREEN,
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_download_one, index, day, chunk_path)
            for index, day, chunk_path in pending
        ]
        for future in as_completed(futures):
            if first_error:
                break
            future.result()

    if first_error:
        raise first_error[0]


def _build_request(config: DownloadConfig, *, day: str, hours: list[str]) -> dict:
    return _build_request_range(config, days=[day], hours=hours)


def _build_request_range(
    config: DownloadConfig,
    *,
    days: Sequence[str],
    hours: list[str],
) -> dict:
    return {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": list(config.variables),
        "pressure_level": list(config.pressure_levels),
        "year": sorted({day[:4] for day in days}),
        "month": sorted({day[5:7] for day in days}),
        "day": sorted({day[8:10] for day in days}),
        "time": hours,
        "area": [config.north, config.west, config.south, config.east],
    }


def _default_chunks_dir(outfile: Path) -> Path:
    return outfile.parent / f"{outfile.stem}.chunks"


def _cds_time_hours(hour_step: int) -> list[str]:
    if hour_step < 1 or hour_step > 24:
        raise typer.BadParameter("hour_step must be between 1 and 24")
    return [f"{h:02d}:00" for h in range(0, 24, hour_step)]


def _chunk_path(chunks_dir: Path, day: str, *, hour_step: int) -> Path:
    suffix = "" if hour_step == 1 else f"_h{hour_step}"
    return chunks_dir / f"era5_{day}{suffix}.nc"


def _is_usable_netcdf(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _time_dimension(dataset) -> str:
    for name in _TIME_DIM_CANDIDATES:
        if name in dataset.dims or name in dataset.coords:
            return name
    raise typer.BadParameter(
        "в NetCDF не найдена ось времени (ожидались valid_time или time)"
    )


def _remove_chunks(chunk_paths: Iterable[Path], chunks_dir: Path) -> None:
    for path in chunk_paths:
        try:
            path.unlink()
        except OSError:
            pass
    try:
        chunks_dir.rmdir()
    except OSError:
        pass


def _date_range(start: str, end: str) -> List[str]:
    start_day = dt.date.fromisoformat(start)
    end_day = dt.date.fromisoformat(end)

    if end_day < start_day:
        raise typer.BadParameter("end date must be >= start date")

    days = []
    cur = start_day

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
