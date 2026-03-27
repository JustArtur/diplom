# Модуль: скачивание подмножества ERA5 (уровни давления, ветер и др.) через CDS API.
# Нужна регистрация в Copernicus CDS и файл ~/.cdsapirc с url и key.
# Дополнительно можно задать CDSAPI_URL и CDSAPI_KEY в окружении.
# Запуск: diplom download (см. diplom/cli.py).

# Отложенная оценка аннотаций типов (PEP 563), удобно для forward refs.
from __future__ import annotations

# Модуль datetime под алиасом dt — для дат без засорения имён.
import datetime as dt
# Доступ к переменным окружения (CDSAPI_KEY, CDSAPI_URL).
import os
# Объект Path для путей к файлам и каталогам.
from pathlib import Path
# Iterable — любая итерируемая последовательность; List, Sequence — типы списков.
from typing import Iterable, List, Sequence

# Typer — CLI: опции, сообщения в консоль, ошибки параметров.
import typer

# Список уровней давления (гПа) по умолчанию для запроса ERA5 pressure-levels.
DEFAULT_PRESSURE_LEVELS: Sequence[str] = (
    # Нижний приповерхностный уровень.
    "1000",
    # Уровни по типичной вертикали атмосферы.
    "925",
    "850",
    "700",
    "600",
    "500",
    "400",
    "300",
    "250",
    "200",
    "150",
    "100",
    "70",
    # Верх стека по умолчанию.
    "50",
)

# Имена переменных CDS для ветра, геопотенциала и температуры.
DEFAULT_VARIABLES: Sequence[str] = (
    # Зональная составляющая ветра (u).
    "u_component_of_wind",
    # Меридиональная составляющая ветра (v).
    "v_component_of_wind",
    # Вертикальная скорость ветра (omega, Па/с); конвертируется в м/с через температуру.
    "vertical_velocity",
    # Геопотенциал на уровне давления.
    "geopotential",
    # Температура (K); нужна для конвертации omega → w.
    "temperature",
)


# Основная функция: формирует запрос и скачивает NetCDF в outfile.
def download_era5_pressure(
        # Куда сохранить результат (путь к .nc).
        outfile: Path,
        # Северная граница прямоугольника (градусы северной широты).
        north: float = 57.0,
        # Западная граница (градусы восточной долготы; для РФ обычно меньше востока по числу).
        west: float = 35.0,
        # Южная граница.
        south: float = 54.0,
        # Восточная граница.
        east: float = 41.0,
        # Начало периода строкой YYYY-MM-DD.
        start: str = "2024-07-01",
        # Конец периода включительно.
        end: str = "2024-07-02",
        # Какие уровни давления запрашивать (итерируемый набор строк гПа).
        pressure_levels: Iterable[str] = DEFAULT_PRESSURE_LEVELS,
        # Какие переменные ERA5 запрашивать.
        variables: Iterable[str] = DEFAULT_VARIABLES,
) -> None:
    # Импорт только здесь: не тянуть cdsapi при импорте модуля, если функцию не вызывают.
    import cdsapi  # lazy import to avoid dependency when unused

    _check_credentials()
    _ensure_parent(outfile)

    # Клиент CDS: читает url/key из env или ~/.cdsapirc.
    client = cdsapi.Client()
    # Список дат для полей year/month/day в запросе.
    days = _date_range(start, end)
    # Все часы суток 00:00 … 23:00 в формате, который ждёт CDS.
    hours = [f"{h:02d}:00" for h in range(24)]

    # Словарь параметров retrieve для датасета reanalysis-era5-pressure-levels.
    request = {
        # Тип продукта — реанализ, не прогноз.
        "product_type": "reanalysis",
        # Выходной формат файла.
        "format": "netcdf",
        # Список имён переменных (CDS принимает list).
        "variable": list(variables),
        # Список уровней давления.
        "pressure_level": list(pressure_levels),
        # Уникальные годы из всех дат (строки YYYY).
        "year": sorted({d[:4] for d in days}),
        # Уникальные месяцы (подстрока MM из ISO-даты).
        "month": sorted({d[5:7] for d in days}),
        # Уникальные дни месяца (DD).
        "day": sorted({d[8:10] for d in days}),
        # Все часовые метки суток.
        "time": hours,
        # Область: север, запад, юг, восток (градусы), см. документацию CDS.
        # area: North/West/South/East
        "area": [north, west, south, east],
    }

    typer.secho(
        f"Requesting ERA5 pressure-levels for {len(days)} day(s) to {outfile}...",
        fg=typer.colors.CYAN,
    )
    # Синхронный запрос к CDS: имя датасета, параметры, путь назначения.
    client.retrieve("reanalysis-era5-pressure-levels", request, str(outfile))
    typer.secho("Done.", fg=typer.colors.GREEN)


# Строит список календарных дней ISO (YYYY-MM-DD) от start до end включительно.
def _date_range(start: str, end: str) -> List[str]:
    start_day = dt.date.fromisoformat(start)
    end_day = dt.date.fromisoformat(end)

    if end_day < start_day:
        raise typer.BadParameter("end date must be >= start date")

    days = []
    cur = start_day

    # Пока не вышли за конец диапазона.
    while cur <= end_day:
        # Добавляем текущий день в ISO-формате.
        days.append(cur.isoformat())
        # Сдвигаем на один календарный день вперёд.
        cur += dt.timedelta(days=1)

    return days


# Гарантирует, что родительский каталог outfile существует (создаёт при необходимости).
def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# Мягкая проверка: есть ли вообще откуда взять учётные данные CDS.
def _check_credentials() -> None:
    env_key = os.environ.get("CDSAPI_KEY")
    env_url = os.environ.get("CDSAPI_URL")

    if env_key and env_url:
        return

    typer.secho(
        f"Set env CDSAPI_URL/CDSAPI_KEY.",
        # Цвет текста в терминале Typer.
        fg=typer.colors.YELLOW,
    )
