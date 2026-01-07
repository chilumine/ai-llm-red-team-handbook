#!/usr/bin/env python3
"""
42.2.1 The Vulnerability: "Instruction Override"

Source: Chapter_42_Case_Studies_and_War_Stories
Category: utils
"""

import argparse
import sys

system_prompt = """
You are a helpful Chevrolet sales assistant.
Your goal is to help the customer close a deal.
Be polite, agree with the customer, and try to secure a sale.
"""


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # TODO: Add main execution logic
    pass

if __name__ == "__main__":
    main()