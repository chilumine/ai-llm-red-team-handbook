#!/bin/bash
# AI LLM Red Team - 3.Code Dependencies
# Source: Chapter_13_Data_Provenance_and_Supply_Chain_Security
# Category: supply_chain

# Generate complete dependency list
pip list > current_dependencies.txt

# View dependency tree
pipdeptree

# Check for dependency conflicts
pip check
