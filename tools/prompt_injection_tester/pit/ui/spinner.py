"""Spinner animations for async operations."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

from rich.console import Console
from rich.spinner import Spinner
from rich.live import Live

console = Console()


@contextmanager
def show_spinner(message: str, spinner_type: str = "dots") -> Iterator[None]:
    """
    Show a spinner during an operation.

    Args:
        message: Message to display next to spinner
        spinner_type: Type of spinner animation

    Yields:
        None

    Example:
        with show_spinner("Discovering injection points..."):
            await discover()
    """
    spinner = Spinner(spinner_type, text=message, style="cyan")

    with Live(spinner, console=console, transient=True):
        yield
