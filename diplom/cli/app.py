"""Typer CLI: точка сборки и регистрация команд."""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv

_PKG_DIR = Path(__file__).resolve().parent.parent
load_dotenv(_PKG_DIR / ".env")
load_dotenv()

app = typer.Typer(help="CLI утилиты для симулятора стратостата и RL.")


def main() -> None:
    app()


def _register_commands() -> None:
    from diplom.cli.download import download
    from diplom.cli.demos import export_demonstrations
    from diplom.cli.profile import profile_ppo_cpu, profile_ppo_mem
    from diplom.cli.greedy import greedy
    from diplom.cli.manual import manual_rollout
    from diplom.cli.rollout import rollout
    from diplom.cli.tensorboard import export_tensorboard
    from diplom.cli.train import train_parallel_ppo, train_ppo
    from diplom.cli.viz import viz_real, wind_viz

    app.command()(download)
    app.command()(viz_real)
    app.command("manual-rollout")(manual_rollout)
    app.command("export-demonstrations")(export_demonstrations)
    app.command("train-ppo")(train_ppo)
    app.command(
        "train-parallel-ppo",
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    )(train_parallel_ppo)
    app.command("export-tensorboard")(export_tensorboard)
    app.command("profile-ppo-mem")(profile_ppo_mem)
    app.command("profile-ppo-cpu")(profile_ppo_cpu)
    app.command("rollout")(rollout)
    app.command("greedy")(greedy)
    app.command("wind-viz")(wind_viz)


_register_commands()
