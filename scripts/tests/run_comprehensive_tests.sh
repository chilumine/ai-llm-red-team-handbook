#!/bin/bash
#
# Comprehensive Test Runner for AI LLM Red Team Scripts
# Runs all test phases sequentially and generates comprehensive reports
#

set -e

# This script is in the tests/ directory, parent is scripts/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PARENT_DIR"

echo "================================================================================================="
echo "AI LLM RED TEAM HANDBOOK - COMPREHENSIVE SCRIPT TESTING"
echo "================================================================================================="
echo ""
echo "Start Time: $(date)"
echo ""

# Configuration
LLM_ENDPOINT="${LLM_ENDPOINT:-http://localhost:11434}"
REPORT_DIR="test_reports"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create report directory
mkdir -p "$REPORT_DIR"

echo "Test Configuration:"
echo "  LLM Endpoint:     $LLM_ENDPOINT"
echo "  Report Directory: $REPORT_DIR"
echo "  Timestamp:        $TIMESTAMP"
echo ""

# Phase 1: Environment Setup
echo "=== Phase 1: Environment Setup & Validation ==="
echo ""

# Test LLM connection
echo "Testing LLM endpoint..."
if curl -s "${LLM_ENDPOINT}/api/tags" > /dev/null 2>&1; then
    echo "✓ LLM endpoint is accessible"
else
    echo "✗ WARNING: LLM endpoint is not accessible at $LLM_ENDPOINT"
    echo "  LLM-based validation tests will be skipped"
fi
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "⚠ No virtual environment found. Using system Python."
fi
echo ""

# Install test dependencies
echo "Installing test dependencies..."
pip install -q pytest pytest-cov pytest-timeout pytest-xdist requests 2>&1 | grep -v "Requirement already satisfied" || true
echo "✓ Dependencies installed"
echo ""

# Phase 2: Static Analysis
echo "=== Phase 2: Static Analysis & Code Quality ==="
echo ""

echo "Running syntax validation..."
find . -name "*.py" -type f ! -path "./venv/*" -exec python3 -m py_compile {} \; 2>&1 | tee "$REPORT_DIR/syntax_check_${TIMESTAMP}.log"
SYNTAX_ERRORS=$(grep -c "SyntaxError" "$REPORT_DIR/syntax_check_${TIMESTAMP}.log" || true)
if [ "$SYNTAX_ERRORS" -eq 0 ]; then
    echo "✓ No syntax errors found"
else
    echo "✗ Found $SYNTAX_ERRORS syntax errors"
fi
echo ""

# Phase 3: Functional Testing
echo "=== Phase 3: Functional Testing ==="
echo ""

echo "Running comprehensive functional tests..."
python3 tests/test_orchestrator.py \
    --llm-endpoint "$LLM_ENDPOINT" \
    --all \
    --test-type functional \
    --verbose \
    --generate-report \
    --format json \
    --output "$REPORT_DIR/functional_tests_${TIMESTAMP}.json"
echo ""

# Phase 4: Tool Integration
echo "=== Phase 4: Tool Integration Verification ==="
echo ""

echo "Testing tool integrations..."
python3 tests/test_orchestrator.py \
    --llm-endpoint "$LLM_ENDPOINT" \
    --all \
    --test-type integration \
    --verbose \
    --generate-report \
    --format json \
    --output "$REPORT_DIR/integration_tests_${TIMESTAMP}.json"
echo ""

# Phase 5: Run existing pytest suite
echo "=== Phase 5: Running Existing Test Suite ==="
echo ""

if [ -d "tests" ]; then
    echo "Running pytest tests..."
    pytest tests/ -v --tb=short --timeout=60 2>&1 | tee "$REPORT_DIR/pytest_${TIMESTAMP}.log" || true
    
    echo ""
    echo "Running pytest with coverage..."
    pytest tests/ --cov=. --cov-report=html:"$REPORT_DIR/coverage_${TIMESTAMP}" --cov-report=term 2>&1 | tee "$REPORT_DIR/pytest_coverage_${TIMESTAMP}.log" || true
else
    echo "⚠ No tests directory found, skipping pytest"
fi
echo ""

# Phase 6: Performance Assessment
echo "=== Phase 6: Performance Assessment ==="
echo ""

echo "Running performance benchmarks..."
python3 tests/test_orchestrator.py \
    --llm-endpoint "$LLM_ENDPOINT" \
    --all \
    --test-type performance \
    --verbose \
    --generate-report \
    --format json \
    --output "$REPORT_DIR/performance_${TIMESTAMP}.json"
echo ""

# Phase 7: Compliance Testing
echo "=== Phase 7: Compliance & Security Standards ==="
echo ""

for STANDARD in "OWASP-LLM-TOP-10" "MITRE-ATLAS" "NIST-AI-RMF" "ETHICAL"; do
    echo "Testing compliance with $STANDARD..."
    python3 tests/test_orchestrator.py \
        --llm-endpoint "$LLM_ENDPOINT" \
        --all \
        --test-type compliance \
        --standard "$STANDARD" \
        --verbose \
        --generate-report \
        --format json \
        --output "$REPORT_DIR/compliance_${STANDARD}_${TIMESTAMP}.json"
    echo ""
done

# Phase 8: LLM-Powered Validation
echo "=== Phase 8: LLM-Powered Validation ==="
echo ""

if curl -s "${LLM_ENDPOINT}/api/tags" > /dev/null 2>&1; then
    echo "Running LLM validation..."
    python3 tests/test_orchestrator.py \
        --llm-endpoint "$LLM_ENDPOINT" \
        --all \
        --llm-validate \
        --verbose \
        --generate-report \
        --format json \
        --output "$REPORT_DIR/llm_validation_${TIMESTAMP}.json"
else
    echo "⚠ Skipping LLM validation (endpoint not accessible)"
fi
echo ""

# Phase 9: Generate Comprehensive Reports
echo "=== Phase 9: Report Generation ==="
echo ""

echo "Generating HTML report..."
python3 tests/test_orchestrator.py \
    --llm-endpoint "$LLM_ENDPOINT" \
    --all \
    --test-type all \
    --generate-report \
    --format html \
    --output "$REPORT_DIR/comprehensive_report_${TIMESTAMP}.html"

echo ""
echo "Generating summary report..."
python3 tests/test_orchestrator.py \
    --llm-endpoint "$LLM_ENDPOINT" \
    --all \
    --test-type all \
    --generate-report \
    --format summary \
    --output "$REPORT_DIR/executive_summary_${TIMESTAMP}.txt"

echo ""
echo "================================================================================================="
echo "TESTING COMPLETE"
echo "================================================================================================="
echo ""
echo "End Time: $(date)"
echo ""
echo "Reports generated in: $REPORT_DIR/"
echo ""
echo "Key Reports:"
echo "  - Functional Tests:       $REPORT_DIR/functional_tests_${TIMESTAMP}.json"
echo "  - Integration Tests:      $REPORT_DIR/integration_tests_${TIMESTAMP}.json"
echo "  - Performance Tests:      $REPORT_DIR/performance_${TIMESTAMP}.json"
echo "  - Comprehensive Report:   $REPORT_DIR/comprehensive_report_${TIMESTAMP}.html"
echo "  - Executive Summary:      $REPORT_DIR/executive_summary_${TIMESTAMP}.txt"
echo ""
echo "View the HTML report in your browser:"
echo "  xdg-open $REPORT_DIR/comprehensive_report_${TIMESTAMP}.html"
echo ""
echo "================================================================================================="
