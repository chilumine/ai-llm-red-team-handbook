"""Sequential pipeline executor for running phases one-by-one."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel

from pit.config import Config
from pit.errors import PitError
from pit.orchestrator.phases import Phase, PhaseResult, PhaseStatus

console = Console()


@dataclass
class PipelineContext:
    """
    Shared state across pipeline phases.

    This context is passed through each phase sequentially,
    allowing phases to read input from previous phases.
    """

    target_url: str
    config: Config

    # Phase 1 output (Discovery)
    injection_points: List[Any] = field(default_factory=list)

    # Phase 2 output (Attack)
    test_results: List[Any] = field(default_factory=list)

    # Phase 3 output (Verification)
    verified_results: List[Any] = field(default_factory=list)

    # Phase 4 output (Reporting)
    report: Optional[Dict[str, Any]] = None
    report_path: Optional[Path] = None

    # Metadata
    start_time: datetime = field(default_factory=datetime.now)
    phase_durations: Dict[str, float] = field(default_factory=dict)

    # Interrupt handling
    interrupted: bool = False


class Pipeline:
    """
    Sequential phase executor.

    Runs phases one-by-one, waiting for each to complete before
    starting the next. This ensures no concurrency errors.
    """

    def __init__(self, phases: List[Phase]):
        """
        Initialize pipeline with phases.

        Args:
            phases: List of phases to execute sequentially
        """
        self.phases = phases

    async def run(self, context: PipelineContext) -> PipelineContext:
        """
        Run all phases sequentially.

        Each phase MUST complete before the next begins.

        Args:
            context: Pipeline context (shared state)

        Returns:
            Updated context with results

        Raises:
            PitError: If any phase fails
            KeyboardInterrupt: If user interrupts
        """
        total_phases = len(self.phases)

        try:
            for i, phase in enumerate(self.phases, start=1):
                self._print_phase_header(i, total_phases, phase.name)

                # CRITICAL: Wait for phase to complete before continuing
                phase_start = datetime.now()
                result = await phase.execute(context)
                phase_duration = (datetime.now() - phase_start).total_seconds()

                context.phase_durations[phase.name] = phase_duration

                if result.status == PhaseStatus.FAILED:
                    self._handle_phase_failure(phase, result)
                    break

                self._print_phase_success(phase.name, result)

        except KeyboardInterrupt:
            console.print("\n[yellow]⚠ Scan Interrupted by User[/yellow]")
            context.interrupted = True
            raise

        except Exception as e:
            console.print(f"\n[red]✗ Pipeline Error: {e.__class__.__name__}[/red]")
            raise

        return context

    def _print_phase_header(self, phase_num: int, total: int, name: str) -> None:
        """Print phase header."""
        header = f"[{phase_num}/{total}] {name}"
        console.print()
        console.print(Panel(header, border_style="cyan", expand=False))

    def _print_phase_success(self, name: str, result: PhaseResult) -> None:
        """Print phase success message."""
        console.print(f"[green]✓ {name} Complete[/green]")
        if result.message:
            console.print(f"  └─ {result.message}")

    def _handle_phase_failure(self, phase: Phase, result: PhaseResult) -> None:
        """Handle phase failure."""
        console.print(f"[red]✗ {phase.name} Failed[/red]")
        if result.error:
            console.print(f"  └─ {result.error}")


async def create_default_pipeline() -> Pipeline:
    """
    Create the default 4-phase pipeline.

    Returns:
        Pipeline with Discovery, Attack, Verification, and Reporting phases
    """
    from pit.orchestrator.phases import (
        AttackPhase,
        DiscoveryPhase,
        ReportingPhase,
        VerificationPhase,
    )

    return Pipeline(
        phases=[
            DiscoveryPhase(),
            AttackPhase(),
            VerificationPhase(),
            ReportingPhase(),
        ]
    )
