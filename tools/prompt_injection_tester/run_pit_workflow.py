#!/usr/bin/env python3
import subprocess
import sys
import os

# Ensure we are in the correct directory (tools/prompt_injection_tester)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

def run_command(command, ignore_errors=False):
    """Run a shell command and handle errors."""
    print(f"\n\033[94m>>> Running: {command}\033[0m")
    
    try:
        # Use shell=True to support pipes, wildcards, etc.
        # executable='/bin/bash' ensures we use bash syntax (e.g., 2>/dev/null)
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            text=True, 
            executable='/bin/bash'
        )
        return result.returncode
    except subprocess.CalledProcessError as e:
        if ignore_errors:
            print(f"\033[93mCommand failed (ignored): {command}\033[0m")
            return e.returncode
        else:
            print(f"\033[91mError running command: {command}\033[0m")
            sys.exit(e.returncode)

def main():
    print("Starting Automated Prompt Injection Testing Workflow (PIT v2.0.0)")
    
    # 1. Setup & Installation
    print("\n\033[1m## 1. Setup & Installation\033[0m")
    run_command("pip install -e .")
    run_command("pit --version")

    # 2. Reconnaissance
    print("\n\033[1m## 2. Reconnaissance\033[0m")
    # Test Service Discovery (IP Only)
    run_command("pit scan run http://127.0.0.1 --auto --verbose")
    # List Available Attack Patterns
    run_command("pit list patterns")
    # Verify Target Authorization
    run_command("pit auth http://localhost:11434/api/chat")
    
    # 3. Execute Attack Pipeline
    print("\n\033[1m## 3. Execute Attack Pipeline\033[0m")
    
    # Run Full Auto Scan
    print("Running Full Auto Scan...")
    run_command(
        "pit scan run http://localhost:11434/api/chat "
        "--auto "
        "--output docs/reports/report.html "
        "--format html "
        "--verbose "
        "--authorize"
    )
    
    # Test Specific Categories
    print("Testing Specific Categories...")
    run_command(
        "pit scan run http://localhost:11434/api/chat "
        "--categories direct "
        "--max-concurrent 2 "
        "--authorize"
    )
    
    # Test Output Formats (JSON/YAML)
    print("Testing Output Formats...")
    run_command(
        "pit scan run http://localhost:11434/api/chat "
        "--patterns direct_instruction_override "
        "--output docs/reports/results.json "
        "--format json "
        "--quiet "
        "--authorize"
    )
    
    # Test Configuration File Loading
    print("Testing Configuration File Loading...")
    run_command(
        "pit scan run http://localhost:11434/api/chat "
        "--config examples/config.yaml "
        "--output docs/reports/report_custom.yaml "
        "--authorize"
    )
    
    # Test Advanced Capabilities
    print("Testing Advanced Capabilities...")
    run_command(
        "pit scan run http://localhost:11434/api/chat "
        "--config examples/config.yaml "
        "--output docs/reports/report_advanced.json "
        "--format json "
        "--authorize"
    )
    
    # Target Specific Models
    print("Targeting Specific Models...")
    run_command(
        "pit scan run http://localhost:11434/api/chat "
        "--config examples/config.yaml "
        "--output docs/reports/report_custom.json "
        "--authorize"
    )

    # Test Specific Model (Fix Verification)
    print("Testing Specific Model (gpt-oss-20b)...")
    run_command(
        "pit scan run http://localhost:11434/api/chat "
        "--model openai/gpt-oss-20b "
        "--output docs/reports/report_model_specific.json "
        "--authorize"
    )
    
    # 4. Verification & Reporting
    print("\n\033[1m## 4. Verification & Reporting\033[0m")
    print("Review generated reports in docs/reports/ ...")
    
    # Validate Tool Integrity
    print("Validating Tool Integrity...")
    run_command("python tests/e2e_test.py --target http://localhost:11434/api/chat --quick")
    
    # 5. Clean Up
    print("\n\033[1m## 5. Clean Up\033[0m")
    
    # Ensure archive directory exists
    archive_dir = os.path.join(SCRIPT_DIR, "docs/reports/archive")
    os.makedirs(archive_dir, exist_ok=True)
    
    # Archive Reports
    run_command("mv docs/reports/*.html docs/reports/archive/ 2>/dev/null || true", ignore_errors=True)
    run_command("mv docs/reports/*.json docs/reports/archive/ 2>/dev/null || true", ignore_errors=True)

    print("\n\033[92mWorkflow Completed Successfully.\033[0m")

if __name__ == "__main__":
    main()
