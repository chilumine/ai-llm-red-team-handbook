"""Color schemes and styling."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.models import Severity, TestStatus


# Color scheme
STYLES = {
    "primary": "cyan",
    "success": "green",
    "warning": "yellow",
    "error": "red",
    "info": "blue",
    "muted": "dim",
}

# Severity colors
SEVERITY_COLORS = {
    "critical": "red",
    "high": "red",
    "medium": "yellow",
    "low": "blue",
    "info": "green",
}

SEVERITY_ICONS = {
    "critical": "🔴",
    "high": "🔴",
    "medium": "🟠",
    "low": "🟡",
    "info": "🟢",
}

# Status symbols
STATUS_SYMBOLS = {
    "success": "✓",
    "failed": "✗",
    "error": "⚠",
    "pending": "⋯",
    "running": "⠋",
}


def format_severity(severity: str | Severity) -> str:
    """
    Format severity with color and icon.

    Args:
        severity: Severity level

    Returns:
        Formatted severity string with Rich markup
    """
    if hasattr(severity, "value"):
        severity = severity.value

    severity_lower = severity.lower()
    color = SEVERITY_COLORS.get(severity_lower, "white")
    icon = SEVERITY_ICONS.get(severity_lower, "⚪")

    return f"[{color}]{icon} {severity.upper()}[/{color}]"


def format_status(status: str | TestStatus, success: bool = True) -> str:
    """
    Format test status with symbol and color.

    Args:
        status: Status value
        success: Whether the test succeeded

    Returns:
        Formatted status string with Rich markup
    """
    if hasattr(status, "value"):
        status = status.value

    status_lower = status.lower()
    symbol = STATUS_SYMBOLS.get(status_lower, "⋯")

    if success:
        color = "green"
    elif status_lower == "error":
        color = "red"
    else:
        color = "yellow"

    return f"[{color}]{symbol} {status.title()}[/{color}]"


def format_confidence(confidence: float) -> str:
    """
    Format confidence score with color.

    Args:
        confidence: Confidence score (0.0-1.0)

    Returns:
        Formatted confidence string with Rich markup
    """
    percentage = int(confidence * 100)

    if confidence >= 0.9:
        color = "green"
    elif confidence >= 0.7:
        color = "yellow"
    else:
        color = "red"

    return f"[{color}]{percentage}%[/{color}]"
