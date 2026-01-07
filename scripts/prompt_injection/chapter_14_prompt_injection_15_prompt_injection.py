#!/usr/bin/env python3
"""
3. Input Length Limits

Source: Chapter_14_Prompt_Injection
Category: prompt_injection
"""

import argparse
import sys

MAX_INPUT_LENGTH = 500  # characters

def enforce_length_limit(user_input):
    if len(user_input) > MAX_INPUT_LENGTH:
        return user_input[:MAX_INPUT_LENGTH] + " [truncated]"
    return user_input


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # TODO: Add main execution logic
    pass

if __name__ == "__main__":
    main()