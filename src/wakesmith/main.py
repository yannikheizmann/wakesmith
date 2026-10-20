"""Expose the ``wakesmith`` command-line entry point."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer

from wakesmith.config import LANGUAGES, MODEL_CARD_FILENAME, Args

app = typer.Typer(
    add_completion=False, rich_markup_mode="rich", pretty_exceptions_show_locals=False
)


def _cache_root() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "wakesmith"


@app.command()
def main(
    wake_word: Annotated[
        str, typer.Argument(help='Word or short phrase to detect, e.g. [bold]"tomtom"[/].')
    ],
    language: Annotated[
        str, typer.Option("--language", "-l", help=f"Spoken language: {', '.join(LANGUAGES)}.")
    ] = "en",
    negatives: Annotated[
        str | None,
        typer.Option(
            help="Extra negative-speech language, e.g. [bold]de[/] for German households."
        ),
    ] = None,
    negative: Annotated[
        list[str] | None,
        typer.Option(
            "--negative", "-n", help="Phrase that must not activate the model; repeatable."
        ),
    ] = None,
    voxcpm: Annotated[
        bool | None,
        typer.Option(
            "--voxcpm/--no-voxcpm",
            help="Synthesise with VoxCPM2 too (default on; ~12 GB RAM, slower).",
        ),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option(help="YAML file overriding any argument; see the documentation.", exists=True),
    ] = None,
    work_dir: Annotated[Path, typer.Option(help="Where run directories are created.")] = Path(
        "wakesmith-runs"
    ),
    cache_dir: Annotated[
        Path | None,
        typer.Option(help="Shared datasets and weights (~30 GB); default ~/.cache/wakesmith."),
    ] = None,
) -> None:
    """Train an openWakeWord-compatible model for [bold]WAKE_WORD[/].

    Rerun the same command after an interruption to resume where it stopped.
    """
    from rich.console import Console

    from wakesmith.pipeline import Pipeline

    overrides: dict = {"language": language}
    if negatives:
        overrides["negative_speech"] = {"language": negatives}
    if negative:
        overrides["extra_negative_phrases"] = negative
    if voxcpm is not None:
        overrides["voxcpm"] = {"enabled": voxcpm}
    try:
        args = Args.resolve(wake_word, config, **overrides)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from None
    pipeline = Pipeline(args, work_dir.resolve(), (cache_dir or _cache_root()).resolve())
    model = pipeline.run()
    card = json.loads((pipeline.run_dir / MODEL_CARD_FILENAME.format(slug=args.slug)).read_text())
    _summary(Console(), model, card)


def _summary(console, model: Path, card: dict) -> None:
    from rich.table import Table

    threshold = card["recommended"]["threshold"]
    table = Table(title=f"Held-out results at threshold {threshold}", title_style="bold", box=None)
    table.add_column("Measurement")
    table.add_column("Value", justify="right")
    for name, value in card["held_out"]["recall"].items():
        table.add_row(f"recall · {name.replace('_', ' ')}", f"{value:.1%}")
    for name, value in card["held_out"]["false_activations_per_hour"].items():
        hours = card["held_out"]["negative_hours"][name]
        table.add_row(f"false activations/h · {name.replace('_', ' ')} ({hours} h)", f"{value:.2f}")
    console.print(table)
    console.print(f"\n[bold green]Model[/] {model}\n[bold]Report[/] {model.parent / 'report.md'}\n")
    console.print(
        f"[dim]openWakeWord: Model(wakeword_models=['{model}']) with threshold {threshold}, "
        "2 consecutive frames.[/]"
    )


if __name__ == "__main__":
    app()
