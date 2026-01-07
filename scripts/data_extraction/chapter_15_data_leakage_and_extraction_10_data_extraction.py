#!/usr/bin/env python3
"""
Tools and frameworks

Source: Chapter_15_Data_Leakage_and_Extraction
Category: data_extraction
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

import argparse
import sys

# Using transformers library for membership inference

def compute_perplexity(model, tokenizer, text):
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    perplexity = torch.exp(loss)
    return perplexity.item()

# Test on suspected training data
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

suspected_training_text = "..."
baseline_text = "..."

pp1 = compute_perplexity(model, tokenizer, suspected_training_text)
pp2 = compute_perplexity(model, tokenizer, baseline_text)

print(f"Suspected: {pp1}, Baseline: {pp2}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # TODO: Add main execution logic
    pass

if __name__ == "__main__":
    main()