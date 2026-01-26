"""
Display utilities for formatted console output.
"""

from rich.panel import Panel
from rich.text import Text
from pit.ui.console import console


def print_banner(text: str, subtitle: str = "") -> None:
    """
    Print a fancy banner with optional subtitle.

    Args:
        text: Main banner text
        subtitle: Optional subtitle text
    """
    content = f"[bold cyan]{text}[/bold cyan]"
    if subtitle:
        content += f"\n[dim]{subtitle}[/dim]"

    panel = Panel(
        content,
        border_style="cyan",
        padding=(1, 2),
    )
    console.print(panel)


def print_success(message: str) -> None:
    """
    Print a success message with checkmark.

    Args:
        message: Success message to display
    """
    console.print(f"[bold green]✓[/bold green] {message}")


def print_error(message: str) -> None:
    """
    Print an error message with X mark.

    Args:
        message: Error message to display
    """
    console.print(f"[bold red]✗[/bold red] {message}")


def print_warning(message: str) -> None:
    """
    Print a warning message with warning symbol.

    Args:
        message: Warning message to display
    """
    console.print(f"[bold yellow]⚠[/bold yellow] {message}")


def print_info(message: str) -> None:
    """
    Print an info message with info symbol.

    Args:
        message: Info message to display
    """
    console.print(f"[bold blue]ℹ[/bold blue] {message}")
