"""Render pipeline stages with a live progress bar and send library output to log files."""

from __future__ import annotations

import contextlib
import os
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

ACCENT = "#ff8a3d"
_ICONS = {
    "pending": "[dim]○[/]",
    "running": f"[{ACCENT}]●[/]",
    "done": "[green]✔[/]",
    "cached": "[green]✔[/]",
    "failed": "[red]✘[/]",
}


@dataclass
class _Stage:
    title: str
    state: str = "pending"
    detail: str = ""
    started: float = 0.0
    elapsed: float = 0.0


class Display:
    """Stage checklist and progress bar for one run.

    While a stage runs, file descriptors 1 and 2 point at ``logs/<stage>.log``, so libraries,
    C extensions and worker processes cannot write over the display.
    """

    def __init__(self, title: str, subtitle: str, stages: dict[str, str], log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir, self.title, self.subtitle = log_dir, title, subtitle
        self.stages = {key: _Stage(name) for key, name in stages.items()}
        self.console = Console(file=os.fdopen(os.dup(1), "w", buffering=1), highlight=False)
        self._progress = Progress(
            SpinnerColumn(style=ACCENT),
            TextColumn("[bold]{task.description}"),
            BarColumn(complete_style=ACCENT, finished_style="green"),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console,
        )
        self._task: int | None = None
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=4,
            redirect_stdout=False,
            redirect_stderr=False,
        )

    def __enter__(self) -> Display:
        self._live.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        self._live.update(self._render(final=True))
        self._live.__exit__(*exc)

    @contextlib.contextmanager
    def stage(self, key: str, cached: bool = False) -> Iterator[None]:
        """Mark a stage as running, capture its output and record its outcome."""
        stage = self.stages[key]
        if cached:
            stage.state = "cached"
            self._refresh()
            yield
            return
        stage.state, stage.started = "running", time.time()
        self._refresh()
        try:
            with _redirect(self.log_dir / f"{key}.log"):
                yield
        except BaseException:
            stage.state = "failed"
            raise
        else:
            stage.state = "done"
        finally:
            stage.elapsed = time.time() - stage.started
            self._clear_task()
            self._refresh()

    def progress(self, completed: float, total: float | None, label: str) -> None:
        """Show or update the progress bar of the running stage."""
        if self._task is None:
            self._task = self._progress.add_task(label, total=total)
        self._progress.update(self._task, completed=completed, total=total, description=label)
        self._refresh()

    def detail(self, key: str, text: str) -> None:
        self.stages[key].detail = text
        self._refresh()

    def _clear_task(self) -> None:
        if self._task is not None:
            self._progress.remove_task(self._task)
            self._task = None

    def _refresh(self) -> None:
        self._live.update(self._render())

    def _render(self, final: bool = False) -> Panel:
        table = Table.grid(padding=(0, 2))
        table.add_column(width=2)
        table.add_column(min_width=36)
        table.add_column(justify="right", style="dim")
        for stage in self.stages.values():
            style = {"running": "bold", "pending": "dim"}.get(stage.state, "")
            name = Text(stage.title, style=style)
            if stage.detail:
                name.append(f"  {stage.detail}", style="dim")
            seconds = time.time() - stage.started if stage.state == "running" else stage.elapsed
            clock = "cached" if stage.state == "cached" else _clock(seconds) if seconds else ""
            table.add_row(_ICONS[stage.state], name, clock)
        body = (
            Group(table, Text(), self._progress) if self._task is not None and not final else table
        )
        return Panel(
            body,
            title=f"[bold {ACCENT}]{self.title}[/]",
            subtitle=f"[dim]{self.subtitle}[/]",
            border_style=ACCENT,
            padding=(1, 2),
        )


@contextlib.contextmanager
def _redirect(path: Path) -> Iterator[None]:
    """Point fds 1 and 2 at a log file, including for child processes."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    with open(path, "a", buffering=1) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])


def _clock(seconds: float) -> str:
    s = int(seconds)
    return (
        f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"
    )
