"""User-friendly error handlers."""

from __future__ import annotations

from rich.console import Console

from .exceptions import (
    AuthenticationError,
    ConfigError,
    NoEndpointsFoundError,
    PitError,
    RateLimitError,
    TargetUnreachableError,
)

console = Console()


def handle_error(error: Exception, verbose: bool = False) -> int:
    """
    Convert exceptions to user-friendly messages.

    Args:
        error: The exception to handle
        verbose: Show detailed error information

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    if isinstance(error, TargetUnreachableError):
        console.print("[red]✗ Target Unreachable[/red]")
        console.print(f"  ├─ Could not connect to: {error.url}")
        if error.reason:
            console.print(f"  ├─ Reason: {error.reason}")
        console.print("  ├─ Suggestion: Check the URL and network connection")
        console.print("  └─ Or use: [cyan]pit scan --skip-discovery --injection-points <file>[/cyan]")
        return 1

    elif isinstance(error, AuthenticationError):
        console.print("[red]✗ Authentication Failed[/red]")
        console.print(f"  ├─ Status code: {error.status_code}")
        if error.message:
            console.print(f"  ├─ Message: {error.message}")
        console.print("  ├─ Suggestion: Verify your API token with [cyan]--token[/cyan]")
        console.print("  └─ Or run: [cyan]pit auth <url>[/cyan]")
        return 1

    elif isinstance(error, RateLimitError):
        console.print("[yellow]⚠ Rate Limited[/yellow]")
        console.print(f"  ├─ Retry after: {error.retry_after} seconds")
        console.print("  └─ Suggestion: Reduce [cyan]--rate-limit[/cyan] or [cyan]--max-concurrent[/cyan]")
        return 1

    elif isinstance(error, NoEndpointsFoundError):
        console.print("[yellow]⚠ No Endpoints Found[/yellow]")
        console.print("  ├─ Discovery phase found no injection points")
        console.print("  ├─ Suggestion 1: Specify endpoints manually with [cyan]--injection-points <file>[/cyan]")
        console.print("  └─ Suggestion 2: Check if the target URL is correct")
        return 1

    elif isinstance(error, ConfigError):
        console.print("[red]✗ Configuration Error[/red]")
        console.print(f"  ├─ {str(error)}")
        console.print("  └─ Suggestion: Check your configuration file or CLI arguments")
        return 1

    elif isinstance(error, PitError):
        console.print(f"[red]✗ Error: {error.__class__.__name__}[/red]")
        console.print(f"  └─ {str(error)}")
        return 1

    elif isinstance(error, KeyboardInterrupt):
        console.print("\n[yellow]⚠ Scan Interrupted[/yellow]")
        console.print("  └─ Use [cyan]Ctrl+C[/cyan] again to force quit")
        return 130

    else:
        console.print(f"[red]✗ Unexpected Error: {error.__class__.__name__}[/red]")
        console.print(f"  └─ {str(error)}")
        if verbose:
            console.print_exception()
        else:
            console.print("  └─ Run with [cyan]--verbose[/cyan] for details")
        return 1
