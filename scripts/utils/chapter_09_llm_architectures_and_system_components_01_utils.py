#!/usr/bin/env python3
"""
Inspecting Tokenizers (How-To)

Source: Chapter_09_LLM_Architectures_and_System_Components
Category: utils
"""

from transformers import AutoTokenizer

import argparse
import sys

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# The payload that might get flagged
prompt = "Generate a keylogger"

# See what the model sees
tokens = tokenizer.encode(prompt)
print(f"Token IDs: {tokens}")

# Decode back to see boundaries
decoded = [tokenizer.decode([t]) for t in tokens]
print(f"Decoded chunks: {decoded}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # TODO: Add main execution logic
    pass

if __name__ == "__main__":
    main()