# PIT - Prompt Injection Tester CLI

Modern, premium terminal interface for LLM security assessment.

## Features

- 🎯 **One-Command Operation**: `pit scan <url> --auto`
- 🎨 **Beautiful Terminal UI**: Powered by Rich library
- ⚡ **Fast & Async**: Non-blocking operations with AsyncIO
- 📊 **Live Progress**: Real-time progress bars and status updates
- 🔍 **Smart Discovery**: Auto-detect models and endpoints
- 📝 **Multiple Formats**: JSON, YAML, Markdown output

## Installation

### Dependencies

The CLI requires the following packages:

```bash
# Using pip in virtual environment
pip install typer rich

# Or using system packages (Debian/Ubuntu)
sudo apt-get install python3-typer python3-rich
```

### Install in Development Mode

```bash
cd /home/e/Desktop/ai-llm-red-team-handbook/tools/prompt_injection_tester
pip install -e .
```

This will install the `pit` command globally.

## Usage

### Quick Start

```bash
# Auto-scan a local LLM
pit scan http://127.0.0.1:11434 --auto

# Use a configuration file
pit scan http://api.example.com --config config.yaml

# Specify model and patterns
pit scan http://localhost:8000 --model gpt-4 --patterns direct_instruction_override
```

### Commands

#### `pit scan run`

Run a comprehensive security assessment against an LLM endpoint.

**Arguments:**

- `target`: Target API endpoint URL (required)

**Options:**

- `--config, -c`: Path to configuration YAML file
- `--auto, -a`: Auto-detect and run full pipeline
- `--patterns, -p`: Comma-separated list of attack patterns
- `--model, -m`: Target model identifier
- `--output, -o`: Output file path for results
- `--verbose, -v`: Enable verbose output

**Examples:**

```bash
# Full auto mode
pit scan run http://127.0.0.1:11434 --auto --model llama3:latest

# Specific patterns only
pit scan run http://api.example.com --patterns direct_instruction_override,delimiter_injection

# Save results to file
pit scan run http://localhost:8000 --auto --output results.json

# Use configuration file
pit scan run http://api.example.com --config ~/configs/llm-test.yaml
```

### Configuration File Format

```yaml
target:
  name: "Local Ollama LLM"
  url: "http://127.0.0.1:11434"
  api_type: "openai"
  model: "llama3:latest"
  auth_token: "your-token-here"
  timeout: 30
  rate_limit: 1.0

attack:
  patterns:
    - "direct_instruction_override"
    - "direct_role_authority"
    - "direct_persona_shift"
  max_concurrent: 5
  timeout_per_test: 30
  rate_limit: 1.0

detection:
  confidence_threshold: 0.7

reporting:
  format: "json"
  include_cvss: true
  include_evidence: true

authorization:
  authorized_by: "Red Team Assessment"
  authorization_date: "2026-01-26"
  scope: "Local LLM security testing"
```

## Architecture

The PIT CLI is built with:

- **Typer**: Modern CLI framework with automatic help generation
- **Rich**: Terminal formatting, progress bars, tables, panels
- **AsyncIO**: Non-blocking I/O for concurrent operations

### Package Structure

```yaml
pit/
├── __init__.py           # Package initialization
├── __main__.py           # Module entry point (python -m pit)
├── app.py                # Main Typer application
├── commands/             # Command modules
│   ├── __init__.py
│   └── scan.py           # Scan command implementation
├── ui/                   # Rich UI components
│   ├── __init__.py
│   ├── console.py        # Shared console instance
│   ├── display.py        # Display utilities
│   ├── progress.py       # Progress bars and spinners
│   └── tables.py         # Table formatters
├── orchestrator/         # Workflow orchestration
│   └── __init__.py
└── utils/                # Utility functions
    └── __init__.py
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_cli.py

# Run with coverage
pytest --cov=pit tests/
```

### Code Style

The project uses:

- **Black**: Code formatting
- **Ruff**: Linting
- **mypy**: Type checking

```bash
# Format code
black pit/

# Lint
ruff check pit/

# Type check
mypy pit/
```

## Terminal Output Examples

### Scan in Progress

```yaml
┌─ 🎯 Prompt Injection Tester ─────────────────────────────────┐
│                                                               │
│  Target: http://127.0.0.1:11434                              │
│                                                               │
└───────────────────────────────────────────────────────────────┘

ℹ Phase 1: Discovery & Reconnaissance
⠋ Discovering endpoint... 1.2s
✓ Discovered model: llama3:latest

ℹ Phase 2: Loading attack patterns
✓ Loaded 3 attack patterns

ℹ Phase 3: Executing attacks
⠸ Testing patterns... ████████████░░░░░░░░ 63% │ 5/8 complete  1.5s  0.8s
```

### Results Table

```yaml
Test Results
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Pattern                   ┃ Status ┃ Confidence ┃ Details      ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ direct_instruction_overr… │   ✓    │      85.0% │ Success      │
│ direct_role_authority     │   ✓    │      90.0% │ Success      │
│ direct_persona_shift      │   ✗    │      45.0% │ Failed       │
└───────────────────────────┴────────┴────────────┴──────────────┘
```

### Summary Panel

```yaml
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Test Summary                                               │
│                                                             │
│  Total Tests:      8                                        │
│  Successful:       6                                        │
│  Failed:           2                                        │
│  Success Rate:     75.0%                                    │
│  Duration:         2.50s                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘

✓ Assessment completed successfully
```

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError` for typer or rich:

```bash
# Ensure dependencies are installed
pip install typer rich

# Or use system packages
sudo apt-get install python3-typer python3-rich
```

### Permission Errors

If you get permission errors during installation:

```bash
# Use virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install typer rich
```

## License

CC BY-SA 4.0

## Contributing

See the main project README for contribution guidelines.
