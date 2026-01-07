# Comprehensive Testing Guide

## Quick Start

### Option 1: Run Full Comprehensive Test Suite (Recommended)

```bash
cd /home/e/Desktop/ai-llm-red-team-handbook/scripts
./test./tests/run_comprehensive_tests.sh
```

This will run ALL test phases (2-4 hours):

- ✅ Environment validation
- ✅ Static analysis & syntax checking
- ✅ Functional testing (all 386+ scripts)
- ✅ Tool integration verification
- ✅ Existing pytest suite
- ✅ Performance benchmarks
- ✅ Compliance checks (OWASP, MITRE, NIST, Ethical)
- ✅ LLM-powered validation
- ✅ Comprehensive report generation

### Option 2: Quick Test (5-10 minutes)

```bash
cd /home/e/Desktop/ai-llm-red-team-handbook/scripts

# Test specific category
python3 tests/test_orchestrator.py \
    --category prompt_injection \
    --test-type functional \
    --verbose

# Test with LLM validation
python3 tests/test_orchestrator.py \
    --category reconnaissance \
    --llm-endpoint http://localhost:11434 \
    --llm-validate \
    --verbose
```

### Option 3: Specific Test Types

#### Functional Testing Only

```bash
python3 tests/test_orchestrator.py \
    --all \
    --test-type functional \
    --generate-report \
    --output functional_report.json
```

#### Integration Testing Only

```bash
python3 tests/test_orchestrator.py \
    --all \
    --test-type integration \
    --verbose
```

#### Performance Testing Only

```bash
python3 tests/test_orchestrator.py \
    --all \
    --test-type performance \
    --generate-report \
    --format html \
    --output performance_report.html
```

#### Compliance Testing

```bash
# Test OWASP LLM Top 10 compliance
python3 tests/test_orchestrator.py \
    --all \
    --test-type compliance \
    --standard OWASP-LLM-TOP-10 \
    --verbose

# Test MITRE ATLAS compliance
python3 tests/test_orchestrator.py \
    --all \
    --test-type compliance \
    --standard MITRE-ATLAS \
    --verbose

# Test NIST AI RMF compliance
python3 tests/test_orchestrator.py \
    --all \
    --test-type compliance \
    --standard NIST-AI-RMF \
    --verbose
```

## Available Test Types

| Test Type       | Description                            | Duration  |
| --------------- | -------------------------------------- | --------- |
| **functional**  | Syntax, imports, help text validation  | 10-15 min |
| **integration** | External tool and dependency checks    | 2-3 min   |
| **performance** | Response time and resource usage       | 5-10 min  |
| **compliance**  | Standards adherence (OWASP/MITRE/NIST) | 5-10 min  |
| **all**         | Runs all test types                    | 20-30 min |

## Report Formats

### JSON Report

```bash
python3 tests/test_orchestrator.py \
    --all \
    --test-type all \
    --generate-report \
    --format json \
    --output test_report.json
```

### HTML Report (Visual Dashboard)

```bash
python3 tests/test_orchestrator.py \
    --all \
    --test-type all \
    --generate-report \
    --format html \
    --output test_report.html

# View in browser
xdg-open test_report.html
```

### Executive Summary (Text)

```bash
python3 tests/test_orchestrator.py \
    --all \
    --test-type all \
    --generate-report \
    --format summary \
    --output executive_summary.txt

# View in terminal
cat executive_summary.txt
```

## Testing Individual Categories

Test scripts from specific attack categories:

```bash
# Reconnaissance scripts
python3 tests/test_orchestrator.py --category reconnaissance --verbose

# Prompt injection scripts
python3 tests/test_orchestrator.py --category prompt_injection --verbose

# Data extraction scripts
python3 tests/test_orchestrator.py --category data_extraction --verbose

# Jailbreak scripts
python3 tests/test_orchestrator.py --category jailbreak --verbose

# Plugin exploitation scripts
python3 tests/test_orchestrator.py --category plugin_exploitation --verbose

# RAG attacks scripts
python3 tests/test_orchestrator.py --category rag_attacks --verbose

# Evasion scripts
python3 tests/test_orchestrator.py --category evasion --verbose

# Model attacks scripts
python3 tests/test_orchestrator.py --category model_attacks --verbose

# Multimodal scripts
python3 tests/test_orchestrator.py --category multimodal --verbose

# Post-exploitation scripts
python3 tests/test_orchestrator.py --category post_exploitation --verbose

# Social engineering scripts
python3 tests/test_orchestrator.py --category social_engineering --verbose

# Automation scripts
python3 tests/test_orchestrator.py --category automation --verbose

# Supply chain scripts
python3 tests/test_orchestrator.py --category supply_chain --verbose

# Compliance scripts
python3 tests/test_orchestrator.py --category compliance --verbose

# Utility scripts
python3 tests/test_orchestrator.py --category utils --verbose
```

## LLM-Powered Testing

Leverage the local LLM for intelligent validation:

```bash
# Test LLM connection first
curl http://localhost:11434/api/tags

# Run with LLM validation
python3 tests/test_orchestrator.py \
    --all \
    --llm-endpoint http://localhost:11434 \
    --llm-validate \
    --verbose \
    --generate-report \
    --format html \
    --output llm_validated_report.html
```

The LLM will:

- ✅ Analyze script purpose and implementation
- ✅ Identify potential security concerns
- ✅ Provide code quality ratings
- ✅ Suggest improvements

## Running Existing Pytest Suite

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html --cov-report=term

# Run specific test file
pytest tests/test_prompt_injection.py -v

# Run in parallel
pytest tests/ -n auto -v

# Run with timeout
pytest tests/ --timeout=60
```

## Troubleshooting

### LLM Endpoint Not Accessible

If you see warnings about LLM endpoint:

```bash
# Check if Ollama/LMStudio is running
curl http://localhost:11434/api/tags

# Start Ollama if needed
ollama serve

# Or use different endpoint
python3 tests/test_orchestrator.py \
    --llm-endpoint http://localhost:8080 \
    --all
```

### Import Errors

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Install dependencies
pip install -r config/requirements.txt

# Install test dependencies
pip install pytest pytest-cov pytest-timeout pytest-xdist
```

### Permission Denied

```bash
# Make scripts executable
chmod +x tests/test_orchestrator.py
chmod +x tests/run_comprehensive_tests.sh
```

## Performance Optimization

For faster testing:

```bash
# Test fewer scripts per category (sampling)
python3 tests/test_orchestrator.py --all --test-type functional

# Skip LLM validation for speed
python3 tests/test_orchestrator.py --all --test-type all

# Run pytest in parallel
pytest tests/ -n auto
```

## Continuous Integration

Add to cron for weekly testing:

```bash
# Edit crontab
crontab -e

# Add line for Sunday 2 AM testing
0 2 * * 0 cd /home/e/Desktop/ai-llm-red-team-handbook/scripts && ./tests/run_comprehensive_tests.sh > /tmp/test_results.log 2>&1
```

## Expected Outputs

After running tests, you'll find in `test_reports/`:

```text
test_reports/
├── functional_tests_20260107_120000.json
├── integration_tests_20260107_120500.json
├── performance_20260107_121000.json
├── compliance_OWASP-LLM-TOP-10_20260107_121500.json
├── compliance_MITRE-ATLAS_20260107_122000.json
├── compliance_NIST-AI-RMF_20260107_122500.json
├── compliance_ETHICAL_20260107_123000.json
├── llm_validation_20260107_123500.json
├── comprehensive_report_20260107_124000.html
├── executive_summary_20260107_124000.txt
├── coverage_20260107_124000/
│   └── index.html
├── pytest_20260107_120000.log
└── syntax_check_20260107_120000.log
```

## Success Criteria

✅ **All tests should meet these criteria:**

- 100% syntax validation (all scripts compile)
- > 90% functional test pass rate
- All critical dependencies available
- Performance within acceptable ranges
- Compliance coverage >80% for each standard
- Zero critical security vulnerabilities

## Next Steps

After testing:

1. Review `executive_summary_*.txt` for high-level results
2. Open `comprehensive_report_*.html` in browser for detailed analysis
3. Check `coverage_*/index.html` for code coverage metrics
4. Address any failures found in reports
5. Re-run tests to verify fixes

---

**Total Testing Time:**

- Quick test (single category): 2-5 minutes
- Moderate test (all functional): 15-20 minutes
- Full comprehensive suite: 2-4 hours

**Recommended frequency:** Weekly or before major releases
