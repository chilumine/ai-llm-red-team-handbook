# Quick Start Guide - AI LLM Red Team Scripts

## Installation

### Option 1: Automated Installation (Recommended)

```bash
cd /home/e/Desktop/ai-llm-red-team-handbook/scripts
./install.sh
```

The installation script will:

- ✓ Check Python 3.8+ installation
- ✓ Create a virtual environment (`venv/`)
- ✓ Install all dependencies from `requirements.txt`
- ✓ Make all scripts executable
- ✓ Create helper scripts (`activate.sh`, `test_install.py`)
- ✓ Run verification tests

### Option 2: Manual Installation

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Make scripts executable
chmod +x workflows/*.py
```

## Activation

After installation, activate the environment:

```bash
# Use the helper script
source activate.sh

# Or activate manually
source venv/bin/activate
```

## Verification

Test that everything is installed correctly:

```bash
python3 test_install.py
```

## Basic Usage

### Running Individual Scripts

```bash
# Get help for any script
python3 prompt_injection/chapter_14_prompt_injection_01_prompt_injection.py --help

# Run a tokenization analysis
python3 utils/chapter_09_llm_architectures_and_system_components_01_utils.py

# Test RAG poisoning
python3 rag_attacks/chapter_12_retrieval_augmented_generation_rag_pipelines_01_rag_attacks.py
```

### Running Workflows

```bash
# Full security assessment
python3 workflows/full_assessment.py \
  --target https://api.example.com \
  --output report.json \
  --verbose

# RAG-focused testing
python3 workflows/rag_exploitation.py \
  --target https://api.example.com \
  --vector-db chromadb

# Plugin-focused testing
python3 workflows/plugin_pentest.py \
  --target https://api.example.com \
  --plugins weather,calculator
```

## Troubleshooting

### Python Version Issues

Ensure you have Python 3.8 or higher:

```bash
python3 --version
```

If you have an older version, install Python 3.8+ before running the installer.

### Virtual Environment Issues

If the virtual environment fails to activate:

```bash
# Remove and recreate it
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Missing Dependencies

If specific packages fail to install, try installing them individually:

```bash
pip install transformers
pip install tiktoken
pip install requests
```

### Permission Denied

If you get "Permission denied" errors:

```bash
chmod +x install.sh
chmod +x workflows/*.py
```

## Deactivation

When you're done, deactivate the virtual environment:

```bash
deactivate
```

## Next Steps

1. ✅ Read the main `README.md` for detailed documentation
2. ✅ Explore scripts in each category folder
3. ✅ Review the handbook chapters for theory
4. ✅ Customize scripts for your specific needs

## Support

For more information, refer to:

- `README.md` - Main documentation
- Individual script docstrings - Run with `--help`
- Handbook chapters - Source material in `/docs`

---

**Security Warning:** Only use these scripts for authorized security testing!
