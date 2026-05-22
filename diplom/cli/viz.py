from __future__ import annotations

import os
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import typer

from diplom.cli.training_options import build_default_app_config, START_TIME_OPTION
from diplom.data.era5_paths import (
    ERA5_PREVIEW_DATA_DIR,
    list_era5_datasets,
    wind_plot_html_path,
)


def viz_real(
    start_time: Optional[datetime] = START_TIME_OPTION,
) -> None:
    """Запуск PyVista-визуализации на реальном ветре."""
    from diplom.viz.pyvista.runner import VisualizationRunner

    app_config = build_default_app_config(start_time=start_time)
    VisualizationRunner().run_real(app_config)


def wind_viz(
    wind_file: Optional[Path] = typer.Option(
        None,
        "--wind-file", "-f",
        help="Один ERA5 NetCDF; без флага — все *.nc из --data-dir",
    ),
    data_dir: Path = typer.Option(
        ERA5_PREVIEW_DATA_DIR,
        "--data-dir",
        help="Каталог с ERA5 NetCDF для просмотра (обрабатывается, если --wind-file не задан)",
    ),
    time: Optional[datetime] = typer.Option(
        None,
        "--time", "-t",
        help=(
            "Временна́я метка среза ERA5 (ISO 8601, например 2024-07-01T12:00:00). "
            "Если не задано — используется первый временной шаг датасета."
        ),
        formats=[
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M",
            "%Y-%m-%d",
        ],
    ),
    output: Path = typer.Option(
        Path("runs/wind"),
        "--output", "-o",
        help="Каталог для HTML-графиков (имя файла = имя датасета без .nc)",
    ),
    stride_lon: int = typer.Option(
        1, "--stride-lon",
        help="Прореживание по долготе (1 = каждая точка, 2 = через одну, ...)",
    ),
    stride_lat: int = typer.Option(
        1, "--stride-lat",
        help="Прореживание по широте",
    ),
    stride_altitude_m: float = typer.Option(
        500.0,
        "--stride-altitude-m",
        help="Шаг по высоте между конусами, м (ветер интерполируется по вертикали)",
    ),
    w_scale: float = typer.Option(
        0.0, "--w-scale",
        help="Масштаб вертикальной компоненты w для наглядности стрелок",
    ),
    open_browser: bool = typer.Option(
        True, "--open/--no-open",
        help="Открыть результат в браузере после сохранения",
    ),
    list_times: bool = typer.Option(
        False, "--list-times",
        help="Вывести все доступные временны́е метки в датасете и выйти",
    ),
    workers: int | None = typer.Option(
        None,
        "--workers", "-j",
        min=1,
        help="Число процессов для параллельной отрисовки. "
        "По умолчанию: min(число новых графиков, число CPU).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Пересоздать HTML-графики, даже если они уже существуют",
    ),
) -> None:
    """Построить интерактивные 3D-графы поля ветра ERA5.

    По умолчанию обходит все ``*.nc`` в ``data/preview/`` и сохраняет HTML в ``runs/wind/``.
    Заголовок графика совпадает с именем датасета; уже существующие файлы пропускаются
    (используйте ``--force`` для пересоздания).

    Примеры:

    \b
      # Все preview-датасеты
      diplom wind-viz

    \b
      # Список доступных временных меток (все датасеты или один файл)
      diplom wind-viz --list-times

    \b
      # Параллельно 4 процесса
      diplom wind-viz -j 4 --stride-lat 2 --stride-lon 2

    \b
      # Один файл, конкретное время
      diplom wind-viz -f data/preview/era5_....nc --time 2024-07-01T12:00:00 --stride-lat 2

    \b
      # Датасеты из каталога обучения
      diplom wind-viz --data-dir data/training

    \b
      # Пересоздать все графики (игнорировать уже существующие)
      diplom wind-viz --force
    """
    from diplom.viz.plotly.wind import (
        WindPlotRenderJob,
        list_available_times,
        render_wind_plots,
    )

    if wind_file is not None:
        dataset_paths = [wind_file]
    else:
        dataset_paths = list_era5_datasets(data_dir)
        if not dataset_paths:
            typer.echo(
                f"[ошибка] В каталоге {data_dir} нет файлов *.nc.\n"
                "Скачайте данные: diplom download --preview (просмотр) "
                "или diplom download (обучение)",
                err=True,
            )
            raise typer.Exit(code=1)

    missing = [p for p in dataset_paths if not p.exists()]
    if missing:
        for path in missing:
            typer.echo(f"[ошибка] Файл ERA5 не найден: {path}", err=True)
        raise typer.Exit(code=1)

    if list_times:
        for path in dataset_paths:
            available = list_available_times(path)
            typer.echo(f"Доступные временны́е метки в {path.name}:")
            for t in available:
                typer.echo(f"  {t}")
        return

    time_ns: int | None = None
    if time is not None:
        time_ns = int(np.datetime64(time).astype("datetime64[ns]").astype(np.int64))

    jobs: list[WindPlotRenderJob] = []
    for dataset_path in dataset_paths:
        plot_path = wind_plot_html_path(dataset_path, output)
        if plot_path.exists() and not force:
            typer.echo(f"Пропуск {dataset_path.name}: график уже есть → {plot_path}")
            continue
        if plot_path.exists() and force:
            typer.echo(f"Пересоздаю {dataset_path.name}: --force → {plot_path}")
        jobs.append(
            WindPlotRenderJob(
                dataset_path=dataset_path,
                output_dir=output,
                time_ns=time_ns,
                stride_lon=stride_lon,
                stride_lat=stride_lat,
                stride_altitude_m=stride_altitude_m,
                w_scale=w_scale,
                force=force,
            )
        )

    if not jobs:
        typer.echo("Новых графиков не создано (все уже есть или нет датасетов).")
        return

    n_workers = workers if workers is not None else min(len(jobs), os.cpu_count() or 1)
    if n_workers > 1:
        typer.echo(f"Параллельная отрисовка: {len(jobs)} график(ов), workers={n_workers}")

    results = render_wind_plots(jobs, workers=n_workers)
    saved_paths: list[Path] = []
    errors: list[str] = []
    steerability_rows: list[tuple[str, float, float, float, float, float | None]] = []

    for result in results:
        for line in result.log_lines:
            typer.echo(line)
        if result.error:
            errors.append(result.error)
        elif result.saved and result.plot_path is not None:
            saved_paths.append(result.plot_path)
        if result.steerability_stats is not None:
            stats = result.steerability_stats
            steerability_rows.append(
                (
                    result.dataset_name,
                    stats.steerability_score,
                    stats.d_local,
                    stats.heading_diversity,
                    stats.curvature_richness,
                    stats.temporal_persistence,
                )
            )

    if len(steerability_rows) > 1:
        typer.echo("")
        typer.echo(
            "Сравнение датасетов по Steerability Score (выше — лучше для RL-обучения):"
        )
        typer.echo(
            f"{'Датасет':<42} {'Score':>6} {'D_loc':>6} {'H':>6} {'C':>6} {'T':>6}"
        )
        for name, score, d_local, heading, curvature, temporal in sorted(
            steerability_rows,
            key=lambda row: row[1],
            reverse=True,
        ):
            t_text = f"{100.0 * temporal:5.1f}" if temporal is not None else "  n/a"
            typer.echo(
                f"{name:<42} {100.0 * score:5.1f} {100.0 * d_local:5.1f} "
                f"{100.0 * heading:5.1f} {100.0 * curvature:5.1f} {t_text}"
            )

    if errors:
        for msg in errors:
            typer.echo(f"[ошибка] {msg}", err=True)
        raise typer.Exit(code=1)

    if not saved_paths:
        typer.echo("Новых графиков не создано (все уже есть или нет датасетов).")
        return

    if open_browser and len(saved_paths) == 1:
        webbrowser.open(saved_paths[0].resolve().as_uri())
