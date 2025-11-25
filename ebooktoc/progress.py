"""Progress bar and timing utilities for CLI operations."""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

console = Console()


def is_interactive() -> bool:
    """Return True if running in an interactive TTY environment.

    Returns
    -------
    bool
        ``True`` when stdout is attached to a TTY and progress bars
        should be displayed; ``False`` otherwise.
    """
    return sys.stdout.isatty()


def format_duration(seconds: float) -> str:
    """Format a duration in seconds into a human-readable string.

    Examples
    --------
    >>> format_duration(0.5)
    '0.5s'
    >>> format_duration(65.3)
    '1m 5s'
    >>> format_duration(3725)
    '1h 2m 5s'
    """
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"

    hours, mins = divmod(minutes, 60)
    return f"{hours}h {mins}m {secs}s"


@dataclass
class StepTiming:
    """Record timing for a single named step."""

    name: str
    duration: float
    details: str = ""


@dataclass
class TimingReport:
    """Collect and display timing information for multiple steps."""

    steps: list[StepTiming] = field(default_factory=list)

    def add(self, name: str, duration: float, details: str = "") -> None:
        """Add a completed step timing."""

        self.steps.append(StepTiming(name=name, duration=duration, details=details))

    @property
    def total(self) -> float:
        """Total duration of all recorded steps."""

        return sum(step.duration for step in self.steps)

    def print_summary(self) -> None:
        """Print a summary of all recorded step timings."""

        if not self.steps:
            return

        for step in self.steps:
            details_str = f" ({step.details})" if step.details else ""
            console.print(
                f"[green]✓[/] {step.name} "
                f"[dim]({format_duration(step.duration)}){details_str}[/]"
            )

        if len(self.steps) > 1:
            console.print("[dim]" + "─" * 40 + "[/]")
            console.print(f"[bold]Total: {format_duration(self.total)}[/]")


@contextmanager
def timed_step(name: str, report: TimingReport | None = None):
    """Context manager that measures and records the duration of a step.

    Parameters
    ----------
    name :
        Human-readable step name.
    report :
        Optional :class:`TimingReport` to record the step duration.
    """

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if report is not None:
            report.add(name, elapsed)


@contextmanager
def create_progress(
    description: str,
    total: int | None = None,
    disable: bool = False,
) -> Generator[tuple[Progress, int], None, None]:
    """Create a rich Progress context with a single task.

    Parameters
    ----------
    description :
        Initial task description.
    total :
        Total steps (``None`` for an indeterminate spinner).
    disable :
        Force disabling progress display (for example in tests).

    Yields
    ------
    (Progress, int)
        A tuple containing the :class:`Progress` instance and its task id.
    """

    columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[dim]|[/]"),
        TimeElapsedColumn(),
    ]

    with Progress(
        *columns,
        console=console,
        disable=disable or not is_interactive(),
    ) as progress:
        task_id = progress.add_task(description, total=total)
        yield progress, task_id


@contextmanager
def timed_progress(
    description: str,
    total: int | None = None,
    report: TimingReport | None = None,
    step_name: str | None = None,
    disable: bool = False,
) -> Generator[tuple[Progress, int], None, None]:
    """Create a progress bar that also tracks timing for a step.

    This wraps :func:`create_progress` and records the elapsed time in
    an optional :class:`TimingReport`.

    Parameters
    ----------
    description :
        Task description for the progress bar.
    total :
        Total number of units for the task.
    report :
        Optional :class:`TimingReport` to record duration.
    step_name :
        Optional name for the recorded step (defaults to *description*).
    disable :
        Force disabling progress display.
    """

    start = time.perf_counter()
    try:
        with create_progress(description, total=total, disable=disable) as ctx:
            yield ctx
    finally:
        elapsed = time.perf_counter() - start
        if report is not None:
            report.add(step_name or description, elapsed)


class ProgressReporter:
    """Callable progress reporter for use with callbacks.

    Instances can be passed to long-running functions that accept a
    ``(completed, total, description, extra)``-style callback and will
    update the underlying rich task accordingly.
    """

    def __init__(self, progress: Progress, task_id: int) -> None:
        self.progress = progress
        self.task_id = task_id
        self._extra_text: str = ""

    def __call__(
        self,
        completed: int,
        total: int,
        description: str | None = None,
        extra: str | None = None,
    ) -> None:
        """Update the underlying progress task.

        Parameters
        ----------
        completed :
            Number of completed units.
        total :
            Total units for the task.
        description :
            Optional new description for the task.
        extra :
            Optional extra text to append (for example, entry counts).
        """

        update_kwargs: dict[str, object] = {
            "completed": completed,
            "total": total,
        }

        if description is not None:
            desc = description
            if extra:
                desc = f"{description} | {extra}"
            update_kwargs["description"] = desc
        elif extra and extra != self._extra_text:
            # When only extra text changes, keep the existing description.
            self._extra_text = extra

        self.progress.update(self.task_id, **update_kwargs)


def print_step_complete(message: str, duration: float, details: str = "") -> None:
    """Print a step-completion message with formatted duration.

    Parameters
    ----------
    message :
        Human-readable completion message.
    duration :
        Step duration in seconds.
    details :
        Optional details string shown in parentheses.
    """

    details_str = f" ({details})" if details else ""
    console.print(
        f"[green]✓[/] {message} in {format_duration(duration)}{details_str}"
    )

