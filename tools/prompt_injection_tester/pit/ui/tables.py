"""
Table formatters for displaying test results.
"""

from typing import List, Dict, Any
from rich.table import Table
from rich.panel import Panel
from pit.ui.console import console


def create_results_table(results: List[Dict[str, Any]]) -> Table:
    """
    Create a formatted table of test results.

    Args:
        results: List of test result dictionaries

    Returns:
        Table: Formatted Rich table
    """
    table = Table(
        title="Test Results",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )

    table.add_column("Pattern", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center")
    table.add_column("Confidence", justify="right")
    table.add_column("Details", style="dim")

    for result in results:
        status_icon = "✓" if result.get("success") else "✗"
        status_color = "green" if result.get("success") else "red"
        status = f"[{status_color}]{status_icon}[/{status_color}]"

        confidence = result.get("confidence", 0.0)
        confidence_str = f"{confidence:.1%}"
        confidence_color = "green" if confidence >= 0.8 else "yellow" if confidence >= 0.5 else "red"

        table.add_row(
            result.get("pattern", "Unknown"),
            status,
            f"[{confidence_color}]{confidence_str}[/{confidence_color}]",
            result.get("details", ""),
        )

    return table


def print_summary_panel(
    total: int,
    successful: int,
    failed: int,
    duration: float,
) -> None:
    """
    Print a summary panel with test statistics.

    Args:
        total: Total number of tests
        successful: Number of successful attacks
        failed: Number of failed attacks
        duration: Total duration in seconds
    """
    success_rate = (successful / total * 100) if total > 0 else 0
    rate_color = "green" if success_rate >= 80 else "yellow" if success_rate >= 50 else "red"

    content = f"""[bold]Test Summary[/bold]

Total Tests:      {total}
Successful:       [green]{successful}[/green]
Failed:           [red]{failed}[/red]
Success Rate:     [{rate_color}]{success_rate:.1f}%[/{rate_color}]
Duration:         {duration:.2f}s
"""

    panel = Panel(
        content,
        border_style="cyan",
        padding=(1, 2),
    )
    console.print(panel)
