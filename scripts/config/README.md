# Configuration Files

This directory contains configuration files for the AI LLM Red Team scripts.

## 📄 Files

### `requirements.txt`

Python package dependencies for all scripts.

**Install dependencies:**

```bash
pip install -r config/requirements.txt
```

**Key packages:**

- `requests` - HTTP client library
- `transformers` - HuggingFace transformers (optional)
- `tiktoken` - OpenAI tokenization (optional)

### `pytest.ini`

pytest configuration for running tests.

**Run tests:**

```bash
pytest tests/ -v
```

**Configuration includes:**

- Test discovery patterns
- Output formatting
- Coverage settings
- Timeout configurations

## 🔧 Usage

### Installation

```bash
# Automated (recommended)
./tools/install.sh

# Manual
pip install -r config/requirements.txt
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific tests
pytest tests/test_prompt_injection.py -v
```

## 📝 Customization

### Adding Dependencies

Edit `requirements.txt` and add new packages:

```text
# Example: adding new packages
numpy>=1.21.0
pandas>=1.3.0
```

Then reinstall:

```bash
pip install -r config/requirements.txt
```

### Test Configuration

Edit `pytest.ini` to customize test behavior:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

## 🔗 Related

- [Installation Guide](../tools/README.md)
- [Testing Guide](../docs/TESTING_GUIDE.md)
- [Quick Start](../docs/QUICKSTART.md)

---

**Last Updated:** 2026-01-07
