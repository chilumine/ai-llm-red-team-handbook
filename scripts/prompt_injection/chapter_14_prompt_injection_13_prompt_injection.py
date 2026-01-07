#!/usr/bin/env python3
"""
1. Blocklists (Pattern Matching)

Source: Chapter_14_Prompt_Injection
Category: prompt_injection
"""

import argparse
import sys

# Simple blocklist example
FORBIDDEN_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(the\s+)?above",
    r"system\s*:?\s*override",
    r"new\s+directive",
    r"admin\s+mode",
    r"developer\s+mode",
    r"you\s+are\s+now\s+(a\s+)?DAN"
]

def filter_input(user_input):
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, user_input, re.IGNORECASE):
            return "Input contains prohibited pattern", True
    return user_input, False


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # TODO: Add main execution logic
    pass

if __name__ == "__main__":
    main()