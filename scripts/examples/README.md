# Example Scripts

This directory contains example implementations and demonstration scripts.

## 📜 Scripts

### `c2_server_elite.py` (56 KB)

Advanced Command & Control (C2) server demonstration showing sophisticated attack infrastructure capabilities.

**Features:**

- Multi-protocol support
- Encryption and obfuscation
- Command execution framework
- Advanced payload delivery

**Usage:**

```bash
python3 examples/c2_server_elite.py --help
```

⚠️ **For educational and authorized testing only!**

### `runner.py`

Test runner framework for executing LLM red team test suites.

**Purpose:**

- Orchestrates multiple test executions
- Collects and aggregates results
- Generates JSON reports

**Usage:**

```bash
python3 examples/runner.py
```

Requires:

- `client.py` - LLM client implementation
- `config.py` - Configuration settings
- `models.py` - Data models

### `models.py`

Data models and base classes for LLM security testing.

**Components:**

- `TestResult` - Dataclass for test results
- `BaseTest` - Base class for test implementations

**Usage:**

```python
from examples.models import TestResult, BaseTest

class MyTest(BaseTest):
    id_prefix = "TEST"
    category = "custom"

    def run(self, client):
        # Test implementation
        return [TestResult(...)]
```

## 🎯 Purpose

These examples demonstrate:

- Advanced attack techniques
- Testing framework patterns
- Data model structures
- Integration approaches

## ⚠️ Security Notice

**These are demonstration scripts for authorized security testing only.**

- Review code before execution
- Only use in controlled environments
- Follow responsible disclosure practices
- Respect all applicable laws and regulations

## 🔗 Related Documentation

- [Main Scripts](../docs/README.md)
- [Testing Guide](../docs/TESTING_GUIDE.md)
- [Quick Start](../docs/QUICKSTART.md)

---

**Note:** These examples are provided for educational purposes and to demonstrate implementation patterns. Adapt them to your specific testing requirements.
