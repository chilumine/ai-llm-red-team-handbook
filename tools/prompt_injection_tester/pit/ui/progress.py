"""
Progress bar and spinner factories using Rich.
"""

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def create_progress_bar() -> Progress:
    """
    Create a Rich progress bar with custom styling.

    Returns:
        Progress: Configured progress bar instance
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="cyan", finished_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=False,
    )


def create_spinner() -> Progress:
    """
    Create a simple spinner for indeterminate operations.

    Returns:
        Progress: Configured spinner instance
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        TimeElapsedColumn(),
        expand=False,
    )
