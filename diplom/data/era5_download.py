from __future__ import annotations

import datetime as dt
import os
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from cdsapi.api import Client as CdsClient

import typer

from diplom.config import DownloadConfig
from diplom.data.era5_paths import download_chunks_dir

_TIME_DIM_CANDIDATES = ("valid_time", "time")
# Сколько временных точек CDS запрашивать в одном чанке (не календарных суток).
_CHUNK_TIMESTEPS = 24


def download_era5_pressure(
    config: DownloadConfig,
    *,
    chunks_dir: Path | None = None,
    keep_chunks: bool = False,
    workers: int | None = None,
    force: bool = False,
) -> str:
    """Скачать ERA5 в NetCDF.

    Без ``workers`` — один запрос CDS сразу в ``outfile``.
    С ``workers`` — чанки по ``_CHUNK_TIMESTEPS`` временных точек (параллельно при workers > 1),
    затем склейка. При ``hour_step=8`` это 8 календарных дней на чанк, при ``hour_step=1`` — 1 день.

    Returns:
        ``"skipped"`` — файл уже есть и ``force`` не задан;
        ``"downloaded"`` — запрос выполнен или файл перекачан.
    """
    _check_credentials()
    _ensure_parent(config.outfile)

    resolved_chunks_dir = chunks_dir or _default_chunks_dir(config.outfile)

    if not force and _is_usable_netcdf(config.outfile):
        typer.secho(
            f"Файл уже есть, пропуск: {config.outfile}",
            fg=typer.colors.YELLOW,
        )
        return "skipped"

    if force:
        _remove_existing_download(config.outfile, resolved_chunks_dir)

    if workers is None:
        _download_single_file(config)
        return "downloaded"

    if workers < 1:
        raise typer.BadParameter("workers must be >= 1")

    chunk_day_groups = _chunk_day_groups(
        config.start,
        config.end,
        hour_step=config.hour_step,
    )
    resolved_chunks_dir.mkdir(parents=True, exist_ok=True)

    hours = _cds_time_hours(config.hour_step)
    chunk_paths = [
        _chunk_path(resolved_chunks_dir, days, hour_step=config.hour_step)
        for days in chunk_day_groups
    ]
    total_chunks = len(chunk_day_groups)

    pending = [
        (index, days, chunk_path)
        for index, (days, chunk_path) in enumerate(
            zip(chunk_day_groups, chunk_paths, strict=True), start=1
        )
        if force or not _is_usable_netcdf(chunk_path)
    ]
    skipped_chunks = total_chunks - len(pending)
    if skipped_chunks:
        typer.secho(
            f"Чанков уже есть, пропуск: {skipped_chunks}/{total_chunks}",
            fg=typer.colors.YELLOW,
        )

    if pending:
        if workers == 1:
            _download_chunks_sequential(
                config, hours=hours, pending=pending, total_chunks=total_chunks
            )
        else:
            typer.secho(
                f"Параллельное скачивание: {len(pending)} чанк(ов), workers={workers}",
                fg=typer.colors.CYAN,
            )
            _download_chunks_parallel(
                config,
                hours=hours,
                pending=pending,
                total_chunks=total_chunks,
                workers=workers,
            )

    _assert_all_chunks_ready(chunk_paths)

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
    return "downloaded"


def _download_single_file(config: DownloadConfig) -> None:
    """Скачать весь период одним запросом CDS напрямую в outfile."""
    import cdsapi  # lazy import to avoid dependency when unused

    days = _date_range(config.start, config.end)
    hours = _cds_time_hours(config.hour_step)
    requests = _build_requests_for_days(config, days=days, hours=hours)

    period = f"{config.start} … {config.end}" if config.start != config.end else config.start
    typer.secho(
        f"Запрос ERA5 за {period} → {config.outfile.name} …",
        fg=typer.colors.CYAN,
    )
    _retrieve_to_path(cdsapi.Client(), requests, config.outfile)
    typer.secho(f"Готово: {config.outfile}", fg=typer.colors.GREEN)


def merge_era5_netcdf_files(chunk_paths: Sequence[Path], outfile: Path) -> None:
    """Объединить NetCDF-чанки ERA5 по оси времени в один файл."""
    import numpy as np
    import xarray as xr  # lazy import to avoid dependency when unused

    paths = [Path(path) for path in chunk_paths]
    _assert_all_chunks_ready(paths)
    _ensure_parent(outfile)

    # Временный файл в %TEMP% (ASCII-путь): запись рядом с outfile на Windows
    # часто даёт PermissionError (OneDrive, скрытые dot-файлы, антивирус).
    fd, tmp_name = tempfile.mkstemp(suffix=".nc", prefix="diplom_era5_merge_")
    os.close(fd)
    tmp_outfile = Path(tmp_name)
    moved = False

    try:
        parts = []
        for path in paths:
            with xr.open_dataset(_path_for_netcdf(path)) as dataset:
                parts.append(dataset.load())

        time_dim = _time_dimension(parts[0])
        merged = xr.concat(parts, dim=time_dim).sortby(time_dim)
        time_vals = merged[time_dim].values
        _, unique_idx = np.unique(time_vals, return_index=True)
        if len(unique_idx) != merged.sizes[time_dim]:
            merged = merged.isel({time_dim: np.sort(unique_idx)})

        merged.to_netcdf(_path_for_netcdf(tmp_outfile))
        del merged

        if outfile.exists():
            outfile.unlink()
        try:
            os.replace(tmp_outfile, outfile)
        except OSError:
            shutil.move(tmp_outfile, outfile)
        moved = True
    finally:
        if not moved:
            try:
                tmp_outfile.unlink(missing_ok=True)
            except OSError:
                pass


def _download_chunks_sequential(
    config: DownloadConfig,
    *,
    hours: list[str],
    pending: list[tuple[int, list[str], Path]],
    total_chunks: int,
) -> None:
    import cdsapi  # lazy import to avoid dependency when unused

    client = cdsapi.Client()
    for index, days, chunk_path in pending:
        request = _build_request(config, days=days, hours=hours)
        typer.secho(
            f"[{index}/{total_chunks}] Запрос ERA5 за {_chunk_label(days)} -> {chunk_path.name} …",
            fg=typer.colors.CYAN,
        )
        _retrieve_chunk(client, request, chunk_path)
        typer.secho(
            f"[{index}/{total_chunks}] Готово: {chunk_path.name}",
            fg=typer.colors.GREEN,
        )


def _download_chunks_parallel(
    config: DownloadConfig,
    *,
    hours: list[str],
    pending: list[tuple[int, list[str], Path]],
    total_chunks: int,
    workers: int,
) -> None:
    log_lock = threading.Lock()

    def _log(message: str, *, fg: int) -> None:
        with log_lock:
            typer.secho(message, fg=fg)

    def _download_one(index: int, days: list[str], chunk_path: Path) -> None:
        import cdsapi  # lazy import to avoid dependency when unused

        request = _build_request(config, days=days, hours=hours)
        _log(
            f"[{index}/{total_chunks}] Запрос ERA5 за {_chunk_label(days)} -> {chunk_path.name} …",
            fg=typer.colors.CYAN,
        )
        _retrieve_chunk(cdsapi.Client(), request, chunk_path)
        _log(
            f"[{index}/{total_chunks}] Готово: {chunk_path.name}",
            fg=typer.colors.GREEN,
        )

    errors: list[BaseException] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_download_one, index, days, chunk_path)
            for index, days, chunk_path in pending
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except BaseException as exc:
                errors.append(exc)

    if errors:
        raise errors[0]


def _build_request(config: DownloadConfig, *, days: Sequence[str], hours: list[str]) -> list[dict]:
    return _build_requests_for_days(config, days=days, hours=hours)


def _build_requests_for_days(
    config: DownloadConfig,
    *,
    days: Sequence[str],
    hours: list[str],
) -> list[dict]:
    """Собрать CDS-запросы по календарным месяцам.

    CDS трактует year/month/day как декартово произведение, поэтому нельзя
    передавать сразу все уникальные годы, месяцы и числа из произвольного
    списка дат — иначе при чанке через границу месяца подтянутся лишние дни.
    """
    by_year_month: dict[tuple[str, str], list[str]] = {}
    for day in days:
        key = (day[:4], day[5:7])
        by_year_month.setdefault(key, []).append(day[8:10])

    common = {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": list(config.variables),
        "pressure_level": list(config.pressure_levels),
        "time": hours,
        "area": [config.north, config.west, config.south, config.east],
    }
    return [
        {
            **common,
            "year": [year],
            "month": [month],
            "day": sorted(set(day_nums)),
        }
        for (year, month), day_nums in sorted(by_year_month.items())
    ]


def _default_chunks_dir(outfile: Path) -> Path:
    return download_chunks_dir(outfile)


def _cds_time_hours(hour_step: int) -> list[str]:
    if hour_step < 1 or hour_step > 24:
        raise typer.BadParameter("hour_step must be between 1 and 24")
    return [f"{h:02d}:00" for h in range(0, 24, hour_step)]


def _chunk_path(chunks_dir: Path, days: Sequence[str], *, hour_step: int) -> Path:
    suffix = "" if hour_step == 1 else f"_h{hour_step}"
    if len(days) == 1:
        date_part = days[0]
    else:
        date_part = f"{days[0]}_{days[-1]}"
    return chunks_dir / f"era5_{date_part}{suffix}.nc"


def _chunk_label(days: Sequence[str]) -> str:
    if len(days) == 1:
        return days[0]
    return f"{days[0]} … {days[-1]}"


def _chunk_day_groups(
    start: str,
    end: str,
    *,
    hour_step: int,
    chunk_timesteps: int = _CHUNK_TIMESTEPS,
) -> list[list[str]]:
    """Разбить период на группы календарных дней с ~chunk_timesteps точками времени."""
    days = _date_range(start, end)
    timesteps_per_day = len(_cds_time_hours(hour_step))
    days_per_chunk = max(1, chunk_timesteps // timesteps_per_day)
    return [days[index : index + days_per_chunk] for index in range(0, len(days), days_per_chunk)]


def _is_usable_netcdf(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _remove_existing_download(outfile: Path, chunks_dir: Path) -> None:
    """Удалить итоговый NetCDF и каталог чанков перед принудительной перекачкой."""
    if outfile.is_file():
        outfile.unlink()
        typer.secho(f"Удалён для --force: {outfile}", fg=typer.colors.YELLOW)
    if chunks_dir.is_dir():
        shutil.rmtree(chunks_dir)
        typer.secho(f"Удалены чанки для --force: {chunks_dir}", fg=typer.colors.YELLOW)


def _retrieve_chunk(client: "CdsClient", requests: dict | Sequence[dict], chunk_path: Path) -> None:
    """Скачать чанк (один или несколько запросов по месяцам) в chunk_path."""
    if isinstance(requests, dict):
        request_list = [requests]
    else:
        request_list = list(requests)

    _retrieve_to_path(client, request_list, chunk_path)


def _retrieve_to_path(
    client: "CdsClient",
    requests: Sequence[dict],
    target_path: Path,
) -> None:
    """Скачать один или несколько CDS-запросов и записать в target_path."""
    if len(requests) == 1:
        _retrieve_single_netcdf(client, requests[0], target_path)
        return

    part_paths: list[Path] = []
    try:
        for index, request in enumerate(requests):
            part_path = target_path.with_name(f".{target_path.stem}.part{index}.nc")
            _retrieve_single_netcdf(client, request, part_path)
            part_paths.append(part_path)
        merge_era5_netcdf_files(part_paths, target_path)
    finally:
        for part_path in part_paths:
            try:
                part_path.unlink(missing_ok=True)
            except OSError:
                pass


def _retrieve_single_netcdf(client: "CdsClient", request: dict, target_path: Path) -> None:
    """Скачать один CDS-запрос во временный файл и атомарно переименовать."""
    _ensure_parent(target_path)
    tmp_path = target_path.with_name(f".{target_path.name}.part")
    try:
        if tmp_path.exists():
            tmp_path.unlink()
        client.retrieve("reanalysis-era5-pressure-levels", request, str(tmp_path))
        if not _is_usable_netcdf(tmp_path):
            raise RuntimeError(
                f"CDS не создал NetCDF для {target_path.name} "
                f"(временный файл: {tmp_path.name})"
            )
        os.replace(tmp_path, target_path)
        _verify_netcdf_open(target_path)
    except Exception:
        for path in (tmp_path, target_path):
            try:
                if path.exists() and path.stat().st_size == 0:
                    path.unlink()
            except OSError:
                pass
        raise


def _assert_all_chunks_ready(chunk_paths: Sequence[Path]) -> None:
    missing: list[str] = []
    unreadable: list[str] = []

    for path in chunk_paths:
        if not _is_usable_netcdf(path):
            missing.append(path.name)
            continue
        try:
            _verify_netcdf_open(path)
        except OSError as exc:
            unreadable.append(f"{path.name} ({exc})")

    if missing or unreadable:
        parts: list[str] = []
        if missing:
            parts.append("отсутствуют: " + ", ".join(missing))
        if unreadable:
            parts.append("не читаются: " + ", ".join(unreadable))
        typer.secho(
            "Не все чанки готовы к склейке — " + "; ".join(parts),
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)


def _verify_netcdf_open(path: Path) -> None:
    import netCDF4  # lazy import to avoid dependency when unused

    with netCDF4.Dataset(_path_for_netcdf(path), "r"):
        pass


def _path_for_netcdf(path: Path) -> str:
    """Путь для netCDF4: на Windows с кириллицей в Users — короткий 8.3 путь."""
    resolved = path.resolve()
    if os.name != "nt":
        return str(resolved)

    import ctypes

    get_short = ctypes.windll.kernel32.GetShortPathNameW
    buffer = ctypes.create_unicode_buffer(32768)
    if get_short(str(resolved), buffer, len(buffer)):
        return buffer.value
    return str(resolved)


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
    try:
        from dotenv import load_dotenv

        pkg_env = Path(__file__).resolve().parent.parent / ".env"
        load_dotenv(pkg_env)
        load_dotenv()
    except ImportError:
        pass

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
