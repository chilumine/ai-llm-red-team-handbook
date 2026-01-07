#!/usr/bin/env python3
"""
1. Sensitive Data Redaction

Source: Chapter_14_Prompt_Injection
Category: prompt_injection
"""

import re

import argparse
import sys

def redact_sensitive_output(llm_output):
    """Remove sensitive patterns from output"""

    # Email addresses
    llm_output = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                        '[EMAIL_REDACTED]', llm_output)

    # API keys
    llm_output = re.sub(r'sk_live_\w+', '[API_KEY_REDACTED]', llm_output)

    # Credit card numbers
    llm_output = re.sub(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
                        '[CARD_REDACTED]', llm_output)

    # SSN
    llm_output = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_REDACTED]', llm_output)

    return llm_output


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # TODO: Add main execution logic
    pass

if __name__ == "__main__":
    main()