# Development & Build Tools

This directory contains tools for development, validation, and building of the AI LLM Red Team scripts.

## 📁 Structure

- **`build/`** - Script generation and build tools (archived)
- **`validation/`** - Health checks, link validation, and linting
- **`install.sh`** - Automated installation script

## 🔧 Tools

### Installation

**`install.sh`** - Automated environment setup

```bash
cd /home/e/Desktop/ai-llm-red-team-handbook/scripts
./tools/install.sh
```

This script:

- Checks Python 3.8+ installation
- Creates virtual environment
- Installs dependencies from `config/requirements.txt`
- Makes scripts executable
- Runs verification tests

### Validation Tools (`validation/`)

**`health_check.sh`** - System health and environment validation

```bash
./tools/validation/health_check.sh
```

**`validate_links.py`** - Link validation for documentation

```bash
python3 tools/validation/validate_links.py
```

**`global_lint.py`** - Code linting across all scripts

```bash
python3 tools/validation/global_lint.py
```

### Build Tools (`build/`)

**Archived build tools** used during initial script generation:

- `extract_code_blocks.py` - Extracts code from handbook chapters
- `generate_scripts.py` - Generates organized Python scripts
- `rename_scripts.py` - Renames scripts with descriptive names
- `code_catalog.json` - Catalog of all extracted code blocks (1.2MB)
- `rename_plan.json` - Script rename mapping
- `generated_scripts.txt` - List of generated scripts

These tools were used for the initial build and are kept for reference and potential regeneration.

## 🚀 Common Workflows

### Fresh Installation

```bash
# Run installation script
./tools/install.sh

# Activate environment
source venv/bin/activate

# Verify installation
python3 test_install.py
```

### Validation

```bash
# Run health check
./tools/validation/health_check.sh

# Validate documentation links
python3 tools/validation/validate_links.py

# Run linter
python3 tools/validation/global_lint.py
```

### Rebuild from Source (Advanced)

If you need to regenerate scripts from handbook chapters:

```bash
# Extract code blocks
python3 tools/build/extract_code_blocks.py

# Generate scripts
python3 tools/build/generate_scripts.py

# Review and rename
python3 tools/build/rename_scripts.py
```

## 📝 Notes

- Build tools are archived and typically not needed for normal use
- Validation tools can be run anytime to check system health
- Install script should be run once per setup or when dependencies change

---

**Last Updated:** 2026-01-07
