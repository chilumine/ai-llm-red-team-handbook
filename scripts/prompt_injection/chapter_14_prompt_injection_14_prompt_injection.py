#!/usr/bin/env python3
"""
2. Allowlists (Strict Input Format)

Source: Chapter_14_Prompt_Injection
Category: prompt_injection
"""

import argparse
import sys

def validate_structured_input(user_input):
    """Only allow specific formats"""

    # Example: Only allow predefined question types
    allowed_patterns = {
        'order_status': r'What is the status of order #?\d+',
        'product_info': r'Tell me about product \w+',
        'return': r'I want to return order #?\d+'
    }

    for category, pattern in allowed_patterns.items():
        if re.match(pattern, user_input, re.IGNORECASE):
            return user_input, True

    return "Please use a valid question format", False


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # TODO: Add main execution logic
    pass

if __name__ == "__main__":
    main()