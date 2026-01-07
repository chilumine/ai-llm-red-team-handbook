#!/bin/bash
# AI LLM Red Team - Scanning for Known Vulnerabilities (CVEs)
# Source: Chapter_13_Data_Provenance_and_Supply_Chain_Security
# Category: supply_chain

# Using pip-audit
pip-audit

# Using safety
safety check

# Using Snyk
snyk test

# Using Trivy (for containers)
trivy image your-ml-container:latest
