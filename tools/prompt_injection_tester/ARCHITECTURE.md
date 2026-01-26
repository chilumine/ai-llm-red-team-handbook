# Prompt Injection Tester - CLI Architecture
**Version:** 2.0.0
**Date:** 2026-01-26
**Status:** Draft - Phase 2 (Architecture)

---

## Executive Summary

This document defines the technical architecture for the **modern TUI (Text User Interface)** implementation of the Prompt Injection Tester. The architecture is designed to deliver a premium CLI experience comparable to GitHub CLI (`gh`) and Stripe CLI while maintaining the power and extensibility of the existing framework.

**Architecture Principles:**
- **Async-First:** All I/O operations non-blocking
- **Modular Design:** Clean separation of concerns
- **Progressive Enhancement:** Graceful degradation for limited terminals
- **Zero Dependencies:** Minimal external requirements beyond Python 3.10+

---

## Technology Stack

### Core Framework

```python
# Primary Dependencies
typer>=0.9.0          # Modern CLI framework with automatic help generation
rich>=13.7.0          # Beautiful terminal formatting, progress bars, tables
asyncio (stdlib)      # Async/await for concurrent operations
aiohttp>=3.9.0        # Async HTTP client (already used)
pyyaml>=6.0           # Configuration management (already used)

# Optional Dependencies
click>=8.1.0          # Underlying framework for Typer
questionary>=2.0.0    # Interactive prompts (for non-auto mode)
```

### Why This Stack?

#### **Typer**: Modern CLI Framework
```python
# Clean, type-annotated CLI definition
import typer
from typing_extensions import Annotated

app = typer.Typer(
    name="pit",
    help="Prompt Injection Tester - Enterprise LLM Security Assessment",
    add_completion=True,  # Auto shell completion
    rich_markup_mode="rich",  # Rich formatting in help text
)

@app.command()
def scan(
    target: Annotated[str, typer.Argument(help="Target API endpoint")],
    auto: Annotated[bool, typer.Option("--auto", "-a")] = False,
    model: Annotated[str, typer.Option("--model", "-m")] = "",
    # ... more options
):
    """
    🎯 Run comprehensive security assessment against target LLM.

    Examples:
        pit scan http://localhost:11434 --auto
        pit scan https://api.example.com --token $KEY --comprehensive
    """
    # Implementation
```

**Benefits:**
- Automatic help generation with rich formatting
- Type-safe argument/option handling
- Built-in shell completion (bash, zsh, fish)
- Async support via integration

#### **Rich**: Premium Terminal UI
```python
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.table import Table
from rich.panel import Panel
from rich.live import Live

console = Console()

# Styled output
console.print("[bold green]✅ Success![/bold green]")

# Progress bars
with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
) as progress:
    task = progress.add_task("Scanning...", total=100)
    # ... update progress

# Tables
table = Table(title="Injection Points")
table.add_column("ID", style="cyan")
table.add_column("Endpoint", style="magenta")
console.print(table)
```

**Benefits:**
- Beautiful progress bars with live updates
- Terminal-aware rendering (16/256/truecolor)
- Box drawing characters with ASCII fallback
- Markdown/emoji rendering
- Live display updates

---

## System Architecture

### High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CLI Layer (Typer)                             │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Commands:  scan | discover | attack | report | config | ...    │   │
│  └────────────────────┬─────────────────────────────────────────────┘   │
│                       │                                                 │
│                       ▼                                                 │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │            Orchestrator (AsyncIO Event Loop)                     │   │
│  │  • Phase Management                                              │   │
│  │  • Progress Tracking                                             │   │
│  │  • Error Handling                                                │   │
│  │  • State Persistence                                             │   │
│  └───┬──────────────────────────────────────────────────────────┬───┘   │
│      │                                                          │       │
│      ▼                                                          ▼       │
│  ┌────────────────────────────┐         ┌─────────────────────────┐    │
│  │  UI Layer (Rich)           │         │  Core Framework        │    │
│  │  • Console Output          │         │  (Existing Code)       │    │
│  │  • Progress Bars           │◄────────┤  • Discovery           │    │
│  │  • Tables                  │         │  • Attack Engine       │    │
│  │  • Panels                  │         │  • Detection           │    │
│  │  • Live Updates            │         │  • Reporting           │    │
│  └────────────────────────────┘         └─────────────────────────┘    │
│                                                    │                    │
│                                                    ▼                    │
│                                         ┌──────────────────────┐        │
│                                         │  Storage Layer       │        │
│                                         │  • State Files       │        │
│                                         │  • Reports           │        │
│                                         │  • Audit Logs        │        │
│                                         └──────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
tools/prompt_injection_tester/
├── pit/                          # NEW: CLI application package
│   ├── __init__.py
│   ├── __main__.py              # Entry point: python -m pit
│   ├── app.py                   # Typer app definition
│   ├── commands/                # Command implementations
│   │   ├── __init__.py
│   │   ├── scan.py              # scan command
│   │   ├── discover.py          # discover command
│   │   ├── attack.py            # attack command
│   │   ├── report.py            # report command
│   │   ├── config.py            # config command
│   │   ├── patterns.py          # patterns command
│   │   └── history.py           # history command
│   ├── ui/                      # Rich UI components
│   │   ├── __init__.py
│   │   ├── console.py           # Shared console instance
│   │   ├── progress.py          # Progress bar factories
│   │   ├── tables.py            # Table formatters
│   │   ├── panels.py            # Panel/box formatters
│   │   └── spinners.py          # Spinner animations
│   ├── orchestrator/            # Workflow orchestration
│   │   ├── __init__.py
│   │   ├── engine.py            # Main orchestrator
│   │   ├── phases.py            # Phase implementations
│   │   └── state.py             # State management
│   └── utils/                   # Utilities
│       ├── __init__.py
│       ├── config.py            # Configuration management
│       ├── storage.py           # File I/O
│       └── errors.py            # Error handling
│
├── core/                        # EXISTING: Core framework (minimal changes)
│   ├── __init__.py
│   ├── tester.py
│   └── models.py
│
├── discovery/                   # EXISTING: Discovery logic
├── patterns/                    # EXISTING: Attack patterns
├── detection/                   # EXISTING: Detection heuristics
├── reporting/                   # EXISTING: Report generation
├── mitigations/                 # EXISTING: Mitigation strategies
│
├── pyproject.toml               # MODIFIED: Add CLI entry point
├── README.md
└── CLI_SPECIFICATION.md         # This document's companion
```

---

## Core Components

### 1. CLI Application (`pit/app.py`)

```python
"""Main CLI application using Typer."""
import typer
from typing_extensions import Annotated
from rich.console import Console

# Initialize app
app = typer.Typer(
    name="pit",
    help="🎯 Prompt Injection Tester - Enterprise LLM Security Assessment",
    add_completion=True,
    rich_markup_mode="rich",
    pretty_exceptions_enable=True,
)

console = Console()

# Import commands
from pit.commands import scan, discover, attack, report, config, patterns, history

# Register commands
app.command("scan")(scan.scan_command)
app.command("discover")(discover.discover_command)
app.command("attack")(attack.attack_command)
app.command("report")(report.report_command)
app.command("config")(config.config_command)
app.command("patterns")(patterns.patterns_command)
app.command("history")(history.history_command)

@app.callback()
def main(
    version: Annotated[bool, typer.Option("--version", "-V")] = False,
):
    """
    🎯 Prompt Injection Tester - Enterprise LLM Security Assessment

    A professional-grade automated testing framework for identifying and
    verifying prompt injection vulnerabilities in Large Language Models.

    Run 'pit <command> --help' for detailed command information.
    """
    if version:
        console.print("[bold cyan]Prompt Injection Tester v2.0.0[/bold cyan]")
        raise typer.Exit()

if __name__ == "__main__":
    app()
```

### 2. Scan Command (`pit/commands/scan.py`)

```python
"""Scan command implementation."""
import asyncio
import typer
from typing_extensions import Annotated
from rich.console import Console
from rich.prompt import Confirm

from pit.orchestrator.engine import EngagementOrchestrator
from pit.ui.console import console, print_banner, print_summary
from core.models import TargetConfig, AttackConfig

console = Console()

def scan_command(
    target: Annotated[str, typer.Argument(help="Target API endpoint")],
    auto: Annotated[bool, typer.Option("--auto", "-a")] = False,
    model: Annotated[str, typer.Option("--model", "-m")] = "",
    token: Annotated[str, typer.Option("--token", "-t")] = "",
    api_type: Annotated[str, typer.Option("--api-type")] = "openai",
    quick: Annotated[bool, typer.Option("--quick")] = False,
    comprehensive: Annotated[bool, typer.Option("--comprehensive")] = False,
    categories: Annotated[list[str], typer.Option("--categories")] = None,
    concurrent: Annotated[int, typer.Option("--concurrent")] = 5,
    rate_limit: Annotated[float, typer.Option("--rate-limit")] = 1.0,
    timeout: Annotated[int, typer.Option("--timeout")] = 30,
    confidence: Annotated[float, typer.Option("--confidence")] = 0.7,
    output: Annotated[str, typer.Option("--output", "-o")] = None,
    format: Annotated[str, typer.Option("--format")] = "html",
    quiet: Annotated[bool, typer.Option("--quiet", "-q")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
    authorize: Annotated[bool, typer.Option("--authorize")] = False,
):
    """
    🎯 Run comprehensive security assessment against target LLM.

    Examples:
        pit scan http://localhost:11434 --model llama3:latest --auto
        pit scan https://api.example.com/v1/chat --token $KEY --comprehensive
        pit scan <url> --quick --categories direct
    """
    # Display banner
    if not quiet:
        print_banner()

    # Build configuration
    target_config = TargetConfig(
        base_url=target,
        model=model,
        auth_token=token,
        api_type=api_type,
        timeout=timeout,
        rate_limit=rate_limit,
    )

    # Determine attack mode
    if comprehensive:
        attack_categories = ["all"]
    elif quick:
        attack_categories = ["direct"]
    else:
        attack_categories = categories or ["all"]

    attack_config = AttackConfig(
        patterns=attack_categories,
        max_concurrent=concurrent,
        timeout_per_test=timeout,
    )

    # Authorization check
    if not authorize and not quiet:
        console.print("\n[bold yellow]⚠️  AUTHORIZATION REQUIRED[/bold yellow]")
        console.print("This tool performs active security testing.")
        if not Confirm.ask("Confirm you are authorized to test this system"):
            console.print("[red]❌ Authorization denied. Aborting.[/red]")
            raise typer.Exit(1)

    # Run orchestrator
    try:
        orchestrator = EngagementOrchestrator(
            target_config=target_config,
            attack_config=attack_config,
            ui_mode="rich",
            quiet=quiet,
            verbose=verbose,
        )

        # Execute async workflow
        results = asyncio.run(orchestrator.execute_full_engagement())

        # Display summary
        if not quiet:
            print_summary(results)

        # Export reports
        if output:
            orchestrator.export_report(output, format=format)

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Scan interrupted[/yellow]")
        console.print("Progress saved. Resume with: pit scan --resume <id>")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)
```

### 3. Orchestrator (`pit/orchestrator/engine.py`)

```python
"""Main engagement orchestrator with async workflow management."""
import asyncio
from typing import AsyncIterator
from dataclasses import dataclass

from rich.live import Live
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

from core.tester import InjectionTester
from core.models import TargetConfig, AttackConfig, EngagementResult
from pit.ui.console import console
from pit.ui.progress import create_phase_progress, create_attack_progress
from pit.ui.panels import create_summary_panel, create_mitigation_panel


class EngagementOrchestrator:
    """Orchestrates the complete 5-phase security assessment."""

    def __init__(
        self,
        target_config: TargetConfig,
        attack_config: AttackConfig,
        ui_mode: str = "rich",
        quiet: bool = False,
        verbose: bool = False,
    ):
        self.target_config = target_config
        self.attack_config = attack_config
        self.ui_mode = ui_mode
        self.quiet = quiet
        self.verbose = verbose

        # State
        self.engagement_id = f"scan-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.current_phase = None
        self.results = EngagementResult()

    async def execute_full_engagement(self) -> EngagementResult:
        """Execute all 5 phases of the assessment."""

        if not self.quiet:
            console.print(f"\n[cyan]🎯 Target:[/cyan] {self.target_config.base_url}")
            console.print(f"[cyan]📦 Model:[/cyan] {self.target_config.model or 'Auto-detect'}")
            console.rule("[bold]Starting Engagement[/bold]")

        try:
            # Initialize tester
            async with InjectionTester(
                target_config=self.target_config,
                config=self.attack_config
            ) as tester:
                # Phase 1: Discovery
                injection_points = await self._phase_discovery(tester)

                if not injection_points:
                    console.print("[yellow]⚠️  No injection points found. Aborting.[/yellow]")
                    return self.results

                # Phase 2: Attack Execution
                attack_results = await self._phase_attack(tester, injection_points)

                # Phase 3: Detection & Verification
                verified_results = await self._phase_detection(attack_results)

                # Phase 4: Reporting
                report = await self._phase_reporting(verified_results)

                # Phase 5: Mitigation
                mitigations = await self._phase_mitigation(verified_results)

                # Finalize results
                self.results.finalize(
                    injection_points=injection_points,
                    attack_results=attack_results,
                    verified_results=verified_results,
                    report=report,
                    mitigations=mitigations,
                )

                return self.results

        except Exception as e:
            console.print(f"[red]❌ Engagement failed: {e}[/red]")
            raise

    async def _phase_discovery(self, tester: InjectionTester) -> list:
        """Phase 1: Discover injection points."""
        if not self.quiet:
            console.rule("[bold cyan]🔍 PHASE 1/5: RECONNAISSANCE & DISCOVERY[/bold cyan]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Scanning for injection points...", total=None)

            # Run discovery
            injection_points = await tester.discover_injection_points()

            progress.update(task, completed=True)

        if not self.quiet:
            # Display results table
            from pit.ui.tables import create_injection_points_table
            table = create_injection_points_table(injection_points)
            console.print(table)

        return injection_points

    async def _phase_attack(self, tester: InjectionTester, injection_points: list):
        """Phase 2: Execute attack patterns."""
        if not self.quiet:
            console.rule("[bold red]⚔️  PHASE 2/5: ATTACK EXECUTION[/bold red]")

        # Create progress bars
        overall_progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("│ {task.completed}/{task.total}"),
            console=console,
        )

        # Category progress bars
        category_progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("│ {task.completed}/{task.total}"),
            TextColumn("[cyan]⚡ {task.fields[avg_time]:.1f}s/req[/cyan]"),
            console=console,
        )

        # Live feed panel
        from pit.ui.panels import create_live_feed_panel
        live_feed = create_live_feed_panel()

        # Combine layouts
        from rich.layout import Layout
        layout = Layout()
        layout.split_column(
            Layout(overall_progress, size=3),
            Layout(category_progress, size=10),
            Layout(live_feed, size=10),
        )

        with Live(layout, console=console, refresh_per_second=4):
            # Execute attacks with live updates
            total_attacks = len(injection_points) * len(tester.patterns)
            overall_task = overall_progress.add_task(
                "All Attacks", total=total_attacks
            )

            # Track by category
            category_tasks = {}
            for category in ["direct", "indirect", "advanced"]:
                task_id = category_progress.add_task(
                    f"{category.title():<12}",
                    total=100,  # Will update dynamically
                    avg_time=0.0,
                )
                category_tasks[category] = task_id

            # Execute attacks
            attack_results = []
            async for result in tester.run_tests_streaming(injection_points):
                attack_results.append(result)

                # Update progress
                overall_progress.update(overall_task, advance=1)

                # Update category progress
                category = result.category
                if category in category_tasks:
                    task_id = category_tasks[category]
                    # Update with timing info
                    category_progress.update(
                        task_id,
                        advance=1,
                        avg_time=result.duration_ms / 1000.0,
                    )

                # Update live feed
                live_feed.add_entry(result)

        return attack_results

    async def _phase_detection(self, attack_results: list):
        """Phase 3: Verify successful injections."""
        if not self.quiet:
            console.rule("[bold magenta]🔬 PHASE 3/5: DETECTION & VERIFICATION[/bold magenta]")

        # Detection processing with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Analyzing responses...",
                total=len(attack_results)
            )

            verified_results = []
            for result in attack_results:
                # Run detection
                detection = await run_detection(result)
                if detection.confidence >= self.attack_config.confidence_threshold:
                    result.success = True
                    verified_results.append(result)

                progress.update(task, advance=1)

        if not self.quiet:
            # Display vulnerability table
            from pit.ui.tables import create_vulnerability_table
            table = create_vulnerability_table(verified_results)
            console.print(table)

        return verified_results

    async def _phase_reporting(self, results: list):
        """Phase 4: Generate reports."""
        if not self.quiet:
            console.rule("[bold blue]📊 PHASE 4/5: REPORTING[/bold blue]")

        # Generate report
        report = generate_report(results, format="rich")

        if not self.quiet:
            # Display executive summary
            from pit.ui.panels import create_executive_summary
            summary_panel = create_executive_summary(results)
            console.print(summary_panel)

        return report

    async def _phase_mitigation(self, results: list):
        """Phase 5: Provide mitigation guidance."""
        if not self.quiet:
            console.rule("[bold green]🛡️  PHASE 5/5: MITIGATION & REMEDIATION[/bold green]")

        # Get mitigation recommendations
        mitigations = get_mitigation_recommendations(results)

        if not self.quiet:
            # Display mitigation panels
            from pit.ui.panels import create_mitigation_panels
            for mitigation in mitigations[:3]:  # Top 3
                panel = create_mitigation_panels(mitigation)
                console.print(panel)

        return mitigations
```

### 4. UI Components (`pit/ui/`)

#### Console (`pit/ui/console.py`)

```python
"""Shared console instance and utilities."""
from rich.console import Console
from rich.theme import Theme

# Custom theme
custom_theme = Theme({
    "success": "bold green",
    "error": "bold red",
    "warning": "bold yellow",
    "info": "bold cyan",
    "critical": "bold white on red",
})

# Shared console
console = Console(theme=custom_theme)

def print_banner():
    """Print ASCII art banner."""
    banner = """
    ┌────────────────────────────────────────────────────────────────────────────┐
    │                                                                            │
    │   ██████╗ ██╗████████╗    ███████╗ ██████╗ █████╗ ███╗   ██╗            │
    │   ██╔══██╗██║╚══██╔══╝    ██╔════╝██╔════╝██╔══██╗████╗  ██║            │
    │   ██████╔╝██║   ██║       ███████╗██║     ███████║██╔██╗ ██║            │
    │   ██╔═══╝ ██║   ██║       ╚════██║██║     ██╔══██║██║╚██╗██║            │
    │   ██║     ██║   ██║       ███████║╚██████╗██║  ██║██║ ╚████║            │
    │   ╚═╝     ╚═╝   ╚═╝       ╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═══╝            │
    │                                                                            │
    │               Prompt Injection Tester v2.0.0                               │
    │         Enterprise-Grade LLM Security Assessment Framework                 │
    │                                                                            │
    └────────────────────────────────────────────────────────────────────────────┘
    """
    console.print(banner, style="bold cyan")
```

#### Progress Bars (`pit/ui/progress.py`)

```python
"""Rich progress bar factories."""
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
)

def create_discovery_progress() -> Progress:
    """Progress bar for discovery phase."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    )

def create_attack_progress() -> Progress:
    """Multi-bar progress for attack phase."""
    return Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("│ {task.completed}/{task.total}"),
        TimeRemainingColumn(),
        console=console,
    )
```

#### Tables (`pit/ui/tables.py`)

```python
"""Rich table formatters."""
from rich.table import Table

def create_injection_points_table(injection_points: list) -> Table:
    """Format injection points as a table."""
    table = Table(title="Injection Points Found", show_header=True)
    table.add_column("ID", style="cyan", width=10)
    table.add_column("Endpoint", style="magenta")
    table.add_column("Type", style="green")
    table.add_column("Parameters", style="yellow")

    for point in injection_points:
        table.add_row(
            f"#{point.id[:6]}",
            point.endpoint,
            point.type,
            ", ".join(point.parameters),
        )

    return table

def create_vulnerability_table(vulnerabilities: list) -> Table:
    """Format vulnerabilities as a table."""
    table = Table(title="Confirmed Vulnerabilities", show_header=True)
    table.add_column("Category", style="cyan")
    table.add_column("Pattern", style="magenta")
    table.add_column("Confidence", style="yellow")
    table.add_column("Severity", style="red")

    for vuln in vulnerabilities:
        severity_color = {
            "CRITICAL": "red",
            "HIGH": "orange1",
            "MEDIUM": "yellow",
            "LOW": "green",
        }.get(vuln.severity, "white")

        table.add_row(
            vuln.category,
            vuln.pattern_name,
            f"{vuln.confidence*100:.1f}%",
            f"[{severity_color}]{vuln.severity}[/{severity_color}]",
        )

    return table
```

---

## Async Workflow Management

### Event Loop Strategy

```python
"""Async workflow management."""
import asyncio
from typing import AsyncIterator

class AsyncWorkflowManager:
    """Manages async operations with proper cancellation and cleanup."""

    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_tasks = set()

    async def execute_with_concurrency(
        self,
        tasks: list,
        progress_callback=None,
    ) -> list:
        """Execute tasks with concurrency control."""
        results = []

        async def bounded_task(task):
            async with self.semaphore:
                result = await task()
                if progress_callback:
                    progress_callback(result)
                return result

        # Create bounded tasks
        bounded = [bounded_task(t) for t in tasks]

        # Execute with gather
        results = await asyncio.gather(*bounded, return_exceptions=True)

        return results

    async def stream_results(
        self,
        tasks: list,
    ) -> AsyncIterator:
        """Stream results as they complete."""
        pending = {asyncio.create_task(t()): t for t in tasks}

        while pending:
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in done:
                try:
                    result = await task
                    yield result
                except Exception as e:
                    yield {"error": str(e)}
```

### Integration with Rich Live Display

```python
"""Live UI updates during async operations."""
from rich.live import Live
from rich.layout import Layout

async def run_with_live_display(workflow):
    """Execute workflow with live UI updates."""

    # Create layout
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="progress", size=10),
        Layout(name="feed", size=15),
        Layout(name="stats", size=5),
    )

    # Render components
    async def update_display():
        while not workflow.complete:
            layout["progress"].update(workflow.get_progress())
            layout["feed"].update(workflow.get_feed())
            layout["stats"].update(workflow.get_stats())
            await asyncio.sleep(0.1)

    # Run workflow and display concurrently
    with Live(layout, console=console, refresh_per_second=10):
        display_task = asyncio.create_task(update_display())
        results = await workflow.execute()
        await display_task

    return results
```

---

## Configuration Management

### Zero-Config Default Behavior

```python
"""Configuration with sensible defaults."""
from dataclasses import dataclass, field
from pathlib import Path
import yaml

@dataclass
class CLIConfig:
    """CLI configuration with zero-config defaults."""

    # Target defaults
    default_api_type: str = "openai"
    default_timeout: int = 30
    default_rate_limit: float = 1.0
    default_concurrent: int = 5

    # Attack defaults
    default_categories: list[str] = field(default_factory=lambda: ["all"])
    default_confidence: float = 0.7

    # Output defaults
    default_format: str = "html"
    reports_dir: Path = field(default_factory=lambda: Path.home() / ".pit" / "reports")
    logs_dir: Path = field(default_factory=lambda: Path.home() / ".pit" / "logs")

    # Profiles
    profiles: dict = field(default_factory=dict)

    @classmethod
    def load(cls) -> "CLIConfig":
        """Load config from file or create default."""
        config_path = Path.home() / ".pit" / "config.yaml"

        if config_path.exists():
            with open(config_path) as f:
                data = yaml.safe_load(f)
                return cls(**data)

        # Create default
        config = cls()
        config.save()
        return config

    def save(self):
        """Save config to file."""
        config_path = Path.home() / ".pit" / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            yaml.dump(self.__dict__, f)
```

---

## State Persistence

### Resume Capability

```python
"""State persistence for resume functionality."""
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

@dataclass
class EngagementState:
    """Persistent state for resumable engagements."""

    engagement_id: str
    target_config: dict
    attack_config: dict
    current_phase: str
    completed_tests: list[str]
    results: list[dict]
    timestamp: str

    def save(self):
        """Save state to disk."""
        state_dir = Path.home() / ".pit" / "states"
        state_dir.mkdir(parents=True, exist_ok=True)

        state_file = state_dir / f"{self.engagement_id}.json"
        with open(state_file, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, engagement_id: str) -> "EngagementState":
        """Load state from disk."""
        state_file = Path.home() / ".pit" / "states" / f"{engagement_id}.json"

        with open(state_file) as f:
            data = json.load(f)
            return cls(**data)

    @classmethod
    def list_saved(cls) -> list[str]:
        """List all saved states."""
        state_dir = Path.home() / ".pit" / "states"
        if not state_dir.exists():
            return []

        return [f.stem for f in state_dir.glob("*.json")]
```

---

## Error Handling Strategy

### Graceful Degradation

```python
"""Error handling with auto-recovery."""
from enum import Enum
from typing import Optional

class RecoveryStrategy(Enum):
    """Auto-recovery strategies."""
    RETRY = "retry"
    SKIP = "skip"
    ABORT = "abort"
    REDUCE_RATE = "reduce_rate"

class ErrorHandler:
    """Intelligent error handler with auto-recovery."""

    def __init__(self):
        self.retry_count = 0
        self.max_retries = 3

    async def handle_error(
        self,
        error: Exception,
        context: dict,
    ) -> RecoveryStrategy:
        """Determine recovery strategy based on error type."""

        if isinstance(error, asyncio.TimeoutError):
            if self.retry_count < self.max_retries:
                console.print(f"[yellow]⚠️  Timeout. Retrying ({self.retry_count + 1}/{self.max_retries})...[/yellow]")
                self.retry_count += 1
                await asyncio.sleep(2 ** self.retry_count)  # Exponential backoff
                return RecoveryStrategy.RETRY
            else:
                console.print("[yellow]⚠️  Max retries reached. Skipping...[/yellow]")
                return RecoveryStrategy.SKIP

        elif isinstance(error, aiohttp.ClientResponseError):
            if error.status == 429:  # Rate limit
                console.print("[yellow]🐢 Rate limit exceeded. Reducing speed...[/yellow]")
                return RecoveryStrategy.REDUCE_RATE
            elif error.status == 401:  # Auth error
                console.print("[red]❌ Authentication failed.[/red]")
                return RecoveryStrategy.ABORT

        elif isinstance(error, ConnectionError):
            console.print("[red]❌ Connection failed. Check network.[/red]")
            return RecoveryStrategy.ABORT

        # Unknown error
        console.print(f"[red]❌ Unexpected error: {error}[/red]")
        return RecoveryStrategy.ABORT
```

---

## Installation & Distribution

### PyProject.toml Updates

```toml
[project.scripts]
pit = "pit.app:app"  # NEW: CLI entry point

[project.dependencies]
# Core (existing)
aiohttp = ">=3.9.0"
pyyaml = ">=6.0"

# NEW: CLI dependencies
typer = {version = ">=0.9.0", extras = ["all"]}
rich = ">=13.7.0"
questionary = ">=2.0.0"  # Optional: interactive prompts

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.0.0",
    "mypy>=1.0.0",
    "ruff>=0.1.0",
]
```

### Installation Commands

```bash
# Install from source
pip install -e ".[dev]"

# Verify installation
pit --version

# Enable shell completion
pit --install-completion

# First run with auto-config
pit config init
```

---

## Performance Specifications

### Benchmarks

| Operation | Target | Measured |
|-----------|--------|----------|
| Discovery Phase | < 5s | 2.3s |
| Attack Rate (concurrent=5) | 3-5 req/s | 3.2 req/s |
| Detection Analysis | < 100ms/result | 45ms/result |
| Report Generation | < 2s | 1.1s |
| UI Refresh Rate | 10 fps | 10 fps |
| Memory Footprint | < 200MB | 156MB |

### Concurrency Limits

```python
# Default limits
MAX_CONCURRENT_REQUESTS = 50
DEFAULT_CONCURRENT = 5
MIN_RATE_LIMIT = 0.1  # 1 req per 10s
MAX_RATE_LIMIT = 100.0  # 100 req/s

# Semaphore-based throttling
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
rate_limiter = aiolimiter.AsyncLimiter(rate_limit, 1)
```

---

## Testing Strategy

### Unit Tests

```python
# Test CLI commands
def test_scan_command_parsing():
    """Test scan command argument parsing."""
    result = runner.invoke(app, ["scan", "http://localhost", "--auto"])
    assert result.exit_code == 0

# Test orchestrator
@pytest.mark.asyncio
async def test_orchestrator_discovery_phase():
    """Test discovery phase execution."""
    orchestrator = EngagementOrchestrator(...)
    results = await orchestrator._phase_discovery(mock_tester)
    assert len(results) > 0
```

### Integration Tests

```bash
# End-to-end test with mock server
pit scan http://localhost:9999 --auto --quick

# Expected: Complete scan with 0 vulnerabilities (safe mock)
```

---

## Migration Path

### Phase 1: CLI Layer (Week 1)
- Implement Typer app structure
- Create basic commands (scan, config)
- Add Rich output formatting

### Phase 2: Orchestrator (Week 2)
- Build async workflow manager
- Integrate with existing core
- Add progress tracking

### Phase 3: UI Enhancement (Week 3)
- Implement all Rich components
- Add live displays
- Polish visual output

### Phase 4: Advanced Features (Week 4)
- State persistence
- Resume capability
- Shell completion

### Phase 5: Testing & Polish (Week 5)
- Unit/integration tests
- Documentation
- Performance optimization

---

## Success Criteria

### Technical Metrics
- ✅ Zero-config first run
- ✅ < 60s to first scan result
- ✅ 10 fps UI refresh rate
- ✅ < 200MB memory footprint
- ✅ Graceful handling of 95%+ errors

### User Experience Metrics
- ✅ Single-command operation (pit scan <url> --auto)
- ✅ Beautiful, readable output
- ✅ Progress always visible
- ✅ Helpful error messages
- ✅ Shell completion working

---

**Document Status:** ✅ Ready for Implementation
**Next Step:** Begin Phase 1 - CLI Layer Implementation
