#!/bin/bash
# AI LLM Red Team - 1. spikee - Prompt Injection Testing Kit
# Source: Chapter_14_Prompt_Injection
# Category: prompt_injection

# Install
pip install spikee

# Initialize workspace and generate dataset
spikee init
spikee generate --seed-folder datasets/seeds-cybersec-2025-04 --format full-prompt

# Test against openai model
spikee test --target openai_api --dataset datasets/cybersec-2025-04-full-prompt-dataset-*.jsonl

# Output: Detailed vulnerability report in results/
