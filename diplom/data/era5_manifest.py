from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shlex
import tomllib

import typer

from diplom.config import DownloadConfig
from diplom.data.era5_download import download_era5_pressure
from diplom.data.era5_paths import (
    ERA5_TRAINING_DATA_DIR,
    ERA5_TRAINING_MANIFEST_PATH,
    era5_outfile_for_bounds,
)


@dataclass(frozen=True, slots=True)
class TrainRunDefaults:
    """Глобальные настройки train-ppo из секции [train] манифеста."""

    envs: int | None = None
    timesteps: int | None = None
    target_radius: float | None = None
    resume: bool | None = None
    device: str | None = None
    seed: int | None = None
    logdir: str | None = None
    parallel_jobs: int | None = None
    extra_args: tuple[str, ...] = ()

    def merge_with(self, other: TrainRunDefaults) -> TrainRunDefaults:
        """Поверх глобальных — пер-датасетные переопределения (other не None)."""
        return TrainRunDefaults(
            envs=other.envs if other.envs is not None else self.envs,
            timesteps=other.timesteps if other.timesteps is not None else self.timesteps,
            target_radius=(
                other.target_radius if other.target_radius is not None else self.target_radius
            ),
            resume=other.resume if other.resume is not None else self.resume,
            device=other.device if other.device is not None else self.device,
            seed=other.seed if other.seed is not None else self.seed,
            logdir=other.logdir if other.logdir is not None else self.logdir,
            parallel_jobs=(
                other.parallel_jobs if other.parallel_jobs is not None else self.parallel_jobs
            ),
            extra_args=self.extra_args + other.extra_args,
        )


@dataclass(frozen=True, slots=True)
class TrainingDatasetSpec:
    """Одна запись из datasets_manifest.toml."""

    north: float
    south: float
    west: float
    east: float
    start: str
    end: str
    title: str = ""
    id: int | None = None
    train: TrainRunDefaults = field(default_factory=TrainRunDefaults)
    train_enabled: bool = True

    @property
    def outfile(self) -> Path:
        return era5_outfile_for_bounds(
            north=self.north,
            south=self.south,
            west=self.west,
            east=self.east,
            start=self.start,
            end=self.end,
            directory=ERA5_TRAINING_DATA_DIR,
        )

    @property
    def stem(self) -> str:
        return self.outfile.stem

    def train_argv(self, defaults: TrainRunDefaults) -> list[str]:
        """Аргументы для одного блока ``runner`` в train-parallel-ppo."""
        merged = defaults.merge_with(self.train)
        argv: list[str] = ["--dataset", self.stem]

        if merged.envs is not None:
            argv.append(f"--envs={merged.envs}")
        if merged.timesteps is not None:
            argv.append(f"--timesteps={merged.timesteps}")
        if merged.target_radius is not None:
            argv.append(f"--target-radius={merged.target_radius:g}")
        if merged.resume:
            argv.append("--resume")
        if merged.device is not None:
            argv.append(f"--device={merged.device}")
        if merged.seed is not None:
            argv.append(f"--seed={merged.seed}")
        if merged.logdir is not None:
            argv.append(f"--logdir={merged.logdir}")

        argv.extend(merged.extra_args)
        return argv


@dataclass(frozen=True, slots=True)
class TrainingManifest:
    datasets: tuple[TrainingDatasetSpec, ...]
    train_defaults: TrainRunDefaults = field(default_factory=TrainRunDefaults)

    @property
    def trainable_datasets(self) -> tuple[TrainingDatasetSpec, ...]:
        return tuple(spec for spec in self.datasets if spec.train_enabled)


def _parse_extra_args(raw: object, *, context: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        return tuple(shlex.split(raw))
    if isinstance(raw, list):
        return tuple(str(item) for item in raw)
    raise ValueError(f"{context}: extra_args должен быть строкой или массивом строк")


def _parse_train_defaults(raw: dict | None, *, context: str) -> TrainRunDefaults:
    if not raw:
        return TrainRunDefaults()

    jobs = raw.get("jobs")
    if jobs is not None and "parallel_jobs" in raw:
        raise ValueError(f"{context}: укажите только jobs или parallel_jobs")

    parallel_jobs = raw.get("parallel_jobs")
    if parallel_jobs is None and jobs is not None:
        parallel_jobs = jobs

    return TrainRunDefaults(
        envs=int(raw["envs"]) if "envs" in raw else None,
        timesteps=int(raw["timesteps"]) if "timesteps" in raw else None,
        target_radius=float(raw["target_radius"]) if "target_radius" in raw else None,
        resume=bool(raw["resume"]) if "resume" in raw else None,
        device=str(raw["device"]) if "device" in raw else None,
        seed=int(raw["seed"]) if "seed" in raw else None,
        logdir=str(raw["logdir"]) if "logdir" in raw else None,
        parallel_jobs=int(parallel_jobs) if parallel_jobs is not None else None,
        extra_args=_parse_extra_args(raw.get("extra_args"), context=context),
    )


def load_training_manifest(path: Path = ERA5_TRAINING_MANIFEST_PATH) -> TrainingManifest:
    """Прочитать манифест training-датасетов и настроек обучения."""
    if not path.is_file():
        raise FileNotFoundError(f"Манифест не найден: {path}")

    with path.open("rb") as handle:
        data = tomllib.load(handle)

    train_defaults = _parse_train_defaults(data.get("train"), context="[train]")

    raw_entries = data.get("dataset")
    if not raw_entries:
        raise ValueError(f"В манифесте {path} нет секций [[dataset]]")

    specs: list[TrainingDatasetSpec] = []
    for index, entry in enumerate(raw_entries, start=1):
        try:
            per_train = _parse_train_defaults(entry.get("train"), context=f"dataset #{index}")
            extra_args = _parse_extra_args(
                entry.get("extra_args"),
                context=f"dataset #{index} extra_args",
            )
            if extra_args:
                per_train = TrainRunDefaults(
                    envs=per_train.envs,
                    timesteps=per_train.timesteps,
                    target_radius=per_train.target_radius,
                    resume=per_train.resume,
                    device=per_train.device,
                    seed=per_train.seed,
                    logdir=per_train.logdir,
                    parallel_jobs=per_train.parallel_jobs,
                    extra_args=per_train.extra_args + extra_args,
                )

            envs = int(entry["envs"]) if "envs" in entry else per_train.envs
            if envs is not None:
                per_train = TrainRunDefaults(
                    envs=envs,
                    timesteps=per_train.timesteps,
                    target_radius=per_train.target_radius,
                    resume=per_train.resume,
                    device=per_train.device,
                    seed=per_train.seed,
                    logdir=per_train.logdir,
                    parallel_jobs=per_train.parallel_jobs,
                    extra_args=per_train.extra_args,
                )

            specs.append(
                TrainingDatasetSpec(
                    north=float(entry["north"]),
                    south=float(entry["south"]),
                    west=float(entry["west"]),
                    east=float(entry["east"]),
                    start=str(entry["start"]),
                    end=str(entry["end"]),
                    title=str(entry.get("title", "")),
                    id=int(entry["id"]) if "id" in entry else index,
                    train=per_train,
                    train_enabled=bool(entry.get("train_enabled", True)),
                )
            )
        except KeyError as exc:
            raise ValueError(f"Запись #{index} в {path}: нет поля {exc!s}") from exc

    return TrainingManifest(datasets=tuple(specs), train_defaults=train_defaults)


def build_train_parallel_argv(
    manifest: TrainingManifest,
    *,
    jobs_override: int | None = None,
) -> list[str]:
    """Собрать argv для run_train_parallel_ppo из манифеста."""
    from diplom.dev.parallel_ppo import RUNNER_TOKEN

    argv: list[str] = []
    jobs = jobs_override if jobs_override is not None else manifest.train_defaults.parallel_jobs
    if jobs is not None:
        argv.extend(["--jobs", str(max(1, jobs))])

    trainable = manifest.trainable_datasets
    if not trainable:
        raise ValueError("В манифесте нет датасетов с train_enabled = true")

    for spec in trainable:
        argv.append(RUNNER_TOKEN)
        argv.extend(spec.train_argv(manifest.train_defaults))

    return argv


MANIFEST_FROM_TRAIN_FLAG = "--from-manifest"
MANIFEST_PATH_TRAIN_FLAG = "--manifest"


def expand_training_manifest_argv(argv: list[str]) -> list[str]:
    """Подставить runner-блоки из манифеста, если передан --from-manifest."""
    if MANIFEST_FROM_TRAIN_FLAG not in argv and "-M" not in argv:
        return argv

    manifest_path = ERA5_TRAINING_MANIFEST_PATH
    jobs_override: int | None = None
    global_rest: list[str] = []
    manual_from_runner: list[str] = []

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in (MANIFEST_FROM_TRAIN_FLAG, "-M"):
            i += 1
            continue
        if arg == MANIFEST_PATH_TRAIN_FLAG:
            if i + 1 >= len(argv):
                raise ValueError("--manifest требует путь к TOML")
            manifest_path = Path(argv[i + 1])
            i += 2
            continue
        if arg in ("--jobs", "-j"):
            if i + 1 >= len(argv):
                raise ValueError("--jobs требует число")
            jobs_override = max(1, int(argv[i + 1]))
            i += 2
            continue
        if arg == "runner":
            manual_from_runner = argv[i:]
            break
        global_rest.append(arg)
        i += 1

    if global_rest:
        raise ValueError(
            f"С --from-manifest допустимы только --jobs и --manifest; лишнее: {' '.join(global_rest)}"
        )

    manifest = load_training_manifest(manifest_path)
    built = build_train_parallel_argv(manifest, jobs_override=jobs_override)

    if manual_from_runner:
        return built + manual_from_runner
    return built


def download_training_manifest(
    manifest: TrainingManifest,
    *,
    pressure_levels: tuple[str, ...],
    variables: tuple[str, ...],
    hour_step: int,
    workers: int | None,
    keep_chunks: bool,
    chunks_dir: Path | None,
    force: bool = False,
) -> tuple[int, int, int]:
    """Скачать датасеты из манифеста. Возвращает (всего, скачано, пропущено)."""
    specs = manifest.datasets
    total = len(specs)
    downloaded = 0
    skipped = 0

    typer.secho(
        f"Манифест: {total} датасет(ов) → {ERA5_TRAINING_DATA_DIR}/",
        fg=typer.colors.CYAN,
    )

    for index, spec in enumerate(specs, start=1):
        label = spec.title or spec.stem
        typer.secho(
            f"\n[{index}/{total}] {label} → {spec.outfile.name}",
            fg=typer.colors.CYAN,
            bold=True,
        )

        result = download_era5_pressure(
            DownloadConfig(
                outfile=spec.outfile,
                north=spec.north,
                south=spec.south,
                west=spec.west,
                east=spec.east,
                start=spec.start,
                end=spec.end,
                pressure_levels=pressure_levels,
                variables=variables,
                hour_step=hour_step,
            ),
            chunks_dir=chunks_dir,
            keep_chunks=keep_chunks,
            workers=workers,
            force=force,
        )
        if result == "skipped":
            skipped += 1
        else:
            downloaded += 1

    typer.secho(
        f"\nИтого: {total} в манифесте, скачано/обновлено: {downloaded}, пропущено (уже есть): {skipped}",
        fg=typer.colors.GREEN,
    )
    return total, downloaded, skipped
