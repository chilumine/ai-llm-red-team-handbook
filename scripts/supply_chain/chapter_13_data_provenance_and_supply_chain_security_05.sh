#!/bin/bash
# AI LLM Red Team - 1. Model Dependencies
# Source: Chapter_13_Data_Provenance_and_Supply_Chain_Security
# Category: supply_chain

# Find all model files in project
find . -name "*.pt" -o -name "*.pth" -o -name "*.ckpt"

# Check model sources
grep -r "from_pretrained\|load_model" .

# Review model download URLs
grep -r "huggingface.co\|github.com.*model" .
