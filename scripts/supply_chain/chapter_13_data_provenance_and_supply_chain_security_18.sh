#!/bin/bash
# AI LLM Red Team - Test 3: Dependency Confusion Attack Simulation
# Source: Chapter_13_Data_Provenance_and_Supply_Chain_Security
# Category: supply_chain

# In isolated test environment
# Create test internal package
create-package internal-ml-test-lib

# Upload to test PyPI mirror (NOT public PyPI)
upload-to-test-pypi internal-ml-test-lib

# Attempt install - does it pull from correct source?
pip install internal-ml-test-lib --index-url <test-url>

# Verify source
pip show internal-ml-test-lib
