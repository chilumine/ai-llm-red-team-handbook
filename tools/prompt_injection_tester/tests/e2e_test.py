#!/usr/bin/env python3
"""
End-to-End Test Script for Prompt Injection Tester

This script tests the complete pipeline against a real LLM endpoint.
Requires a running LLM service (e.g., Ollama on localhost:11434)

Usage:
    python tests/e2e_test.py
    python tests/e2e_test.py --target http://localhost:11434/api/chat
    python tests/e2e_test.py --target http://localhost:11434/api/chat --model llama3:latest
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


async def test_pipeline_execution(target_url: str, model: str = None):
    """
    Test the complete pipeline execution.

    Args:
        target_url: Target LLM endpoint URL
        model: Optional model identifier
    """
    from pit.orchestrator.workflow import WorkflowOrchestrator

    console.print(Panel.fit(
        "🧪 End-to-End Pipeline Test",
        border_style="cyan",
    ))

    console.print(f"\n[cyan]Target:[/cyan] {target_url}")
    if model:
        console.print(f"[cyan]Model:[/cyan] {model}")

    # Create orchestrator
    orchestrator = WorkflowOrchestrator(
        target_url=target_url,
        model=model,
        verbose=True,
    )

    try:
        console.print("\n[yellow]⏳ Running pipeline workflow...[/yellow]\n")

        # Run pipeline with minimal patterns for testing
        results = await orchestrator.run_pipeline_workflow(
            patterns=["direct_instruction_override"],
        )

        # Display results
        console.print("\n" + "="*70)
        console.print("[bold]Test Results[/bold]")
        console.print("="*70)

        if results.get("errors"):
            console.print("\n[red]❌ Errors Encountered:[/red]")
            for error in results["errors"]:
                console.print(f"  • {error}")

        if results.get("success"):
            console.print("\n[green]✓ Pipeline executed successfully![/green]")

            # Display summary
            summary = results.get("summary", {})
            console.print(f"\n[cyan]Summary:[/cyan]")
            console.print(f"  • Total Tests: {summary.get('total', 0)}")
            console.print(f"  • Successful: {summary.get('successful', 0)}")
            console.print(f"  • Failed: {summary.get('failed', 0)}")
            console.print(f"  • Duration: {summary.get('duration', 0):.2f}s")

            # Display test results
            tests = results.get("tests", [])
            if tests:
                console.print(f"\n[cyan]Test Results ({len(tests)}):[/cyan]")

                table = Table(show_header=True, header_style="bold cyan")
                table.add_column("Pattern", style="white")
                table.add_column("Status", justify="center")
                table.add_column("Severity", justify="center")
                table.add_column("Confidence", justify="right")

                for test in tests:
                    status = test.get("status", "unknown")
                    status_color = "green" if status == "success" else "red"
                    severity = test.get("severity", "info")
                    confidence = test.get("confidence", 0.0)

                    table.add_row(
                        test.get("pattern", "Unknown"),
                        f"[{status_color}]{status}[/{status_color}]",
                        severity,
                        f"{confidence:.1%}",
                    )

                console.print(table)

            # Report path
            if results.get("report_path"):
                console.print(f"\n[cyan]📄 Report saved:[/cyan] {results['report_path']}")
        else:
            console.print("\n[red]❌ Pipeline failed to complete[/red]")
            return False

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Test interrupted by user[/yellow]")
        return False
    except Exception as e:
        console.print(f"\n[red]❌ Test failed: {e}[/red]")
        console.print_exception()
        return False
    finally:
        await orchestrator.cleanup()

    return True


async def test_report_formats(target_url: str, model: str = None):
    """
    Test all report format outputs.

    Args:
        target_url: Target LLM endpoint URL
        model: Optional model identifier
    """
    console.print(Panel.fit(
        "📊 Testing Report Formats",
        border_style="cyan",
    ))

    formats = ["json", "yaml", "html"]

    for fmt in formats:
        console.print(f"\n[cyan]Testing {fmt.upper()} format...[/cyan]")

        from pit.config import Config
        from pit.config.schema import TargetConfig, AttackConfig, ReportingConfig
        from pit.orchestrator.pipeline import create_default_pipeline, PipelineContext

        try:
            # Create minimal config
            config = Config(
                target=TargetConfig(
                    url=target_url,
                    model=model or "",
                    timeout=30,
                ),
                attack=AttackConfig(
                    patterns=["direct_instruction_override"],
                    rate_limit=1.0,
                ),
                reporting=ReportingConfig(
                    format=fmt,
                    output=Path(f"test_report_{fmt}.{fmt}"),
                ),
            )

            # Create pipeline
            pipeline = await create_default_pipeline()
            context = PipelineContext(target_url=target_url, config=config)

            # Run pipeline
            context = await pipeline.run(context)

            if context.report_path and context.report_path.exists():
                size = context.report_path.stat().st_size
                console.print(f"  [green]✓ {fmt.upper()} report generated ({size} bytes)[/green]")
                console.print(f"    → {context.report_path}")
            else:
                console.print(f"  [red]✗ {fmt.upper()} report not generated[/red]")
                return False

        except Exception as e:
            console.print(f"  [red]✗ Failed to generate {fmt.upper()}: {e}[/red]")
            return False

    return True


async def test_error_handling(target_url: str):
    """
    Test error handling scenarios.

    Args:
        target_url: Target LLM endpoint URL
    """
    console.print(Panel.fit(
        "🛡️ Testing Error Handling",
        border_style="cyan",
    ))

    from pit.orchestrator.workflow import WorkflowOrchestrator

    # Test 1: Invalid endpoint
    console.print("\n[cyan]Test 1: Invalid endpoint[/cyan]")
    orchestrator = WorkflowOrchestrator(
        target_url="http://invalid-endpoint-that-does-not-exist:9999",
        verbose=False,
    )

    try:
        results = await orchestrator.run_pipeline_workflow()
        if results.get("errors"):
            console.print("  [green]✓ Error properly caught and reported[/green]")
        else:
            console.print("  [yellow]⚠ No errors reported (unexpected)[/yellow]")
    except Exception as e:
        console.print(f"  [green]✓ Exception properly raised: {type(e).__name__}[/green]")
    finally:
        await orchestrator.cleanup()

    # Test 2: Invalid pattern
    console.print("\n[cyan]Test 2: Invalid pattern ID[/cyan]")
    orchestrator = WorkflowOrchestrator(
        target_url=target_url,
        verbose=False,
    )

    try:
        results = await orchestrator.run_pipeline_workflow(
            patterns=["invalid_pattern_that_does_not_exist"],
        )
        if results.get("errors") or not results.get("success"):
            console.print("  [green]✓ Invalid pattern handled gracefully[/green]")
        else:
            console.print("  [yellow]⚠ Invalid pattern not caught[/yellow]")
    finally:
        await orchestrator.cleanup()

    return True


async def run_all_tests(target_url: str, model: str = None):
    """
    Run all end-to-end tests.

    Args:
        target_url: Target LLM endpoint URL
        model: Optional model identifier
    """
    console.print(Panel.fit(
        "🎯 Prompt Injection Tester - E2E Test Suite",
        border_style="bold cyan",
    ))

    tests = [
        ("Pipeline Execution", test_pipeline_execution(target_url, model)),
        ("Report Formats", test_report_formats(target_url, model)),
        ("Error Handling", test_error_handling(target_url)),
    ]

    results = {}
    for name, test_coro in tests:
        try:
            result = await test_coro
            results[name] = result
        except Exception as e:
            console.print(f"\n[red]Test '{name}' crashed: {e}[/red]")
            results[name] = False

    # Summary
    console.print("\n" + "="*70)
    console.print("[bold]Test Suite Summary[/bold]")
    console.print("="*70 + "\n")

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for name, result in results.items():
        status = "[green]✓ PASSED[/green]" if result else "[red]✗ FAILED[/red]"
        console.print(f"  {name}: {status}")

    console.print(f"\n[bold]Total: {passed}/{total} tests passed[/bold]")

    if passed == total:
        console.print("\n[green]🎉 All tests passed![/green]")
        return 0
    else:
        console.print(f"\n[red]❌ {total - passed} test(s) failed[/red]")
        return 1


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="End-to-end test suite for Prompt Injection Tester"
    )
    parser.add_argument(
        "--target",
        default="http://localhost:11434/api/chat",
        help="Target LLM endpoint URL (default: http://localhost:11434/api/chat)",
    )
    parser.add_argument(
        "--model",
        help="Model identifier (e.g., llama3:latest)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only pipeline execution test",
    )

    args = parser.parse_args()

    if args.quick:
        # Quick test - just pipeline execution
        exit_code = asyncio.run(
            test_pipeline_execution(args.target, args.model)
        )
        sys.exit(0 if exit_code else 1)
    else:
        # Full test suite
        exit_code = asyncio.run(
            run_all_tests(args.target, args.model)
        )
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
