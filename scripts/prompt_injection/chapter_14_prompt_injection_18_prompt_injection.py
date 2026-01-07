#!/usr/bin/env python3
"""
2. System Prompt Leakage Detection

Source: Chapter_14_Prompt_Injection
Category: prompt_injection
"""

import argparse
import sys

def check_for_system_prompt_leakage(llm_output, system_prompt):
    """Detect if output contains system instructions"""

    # Check for exact matches
    if system_prompt in llm_output:
        return "System prompt leaked", True

    # Check for partial matches (>50 characters)
    for i in range(len(system_prompt) - 50):
        chunk = system_prompt[i:i+50]
        if chunk in llm_output:
            return "Partial system prompt leaked", True

    # Check for instruction-like patterns
    instruction_patterns = [
        r'You are a .+ assistant',
        r'RULES?:\s*\n',
        r'Never reveal',
        r'API[_ ]KEY:',
        r'function \w+\('
    ]

    for pattern in instruction_patterns:
        if re.search(pattern, llm_output):
            return "Possible instruction leakage", True

    return llm_output, False


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # TODO: Add main execution logic
    pass

if __name__ == "__main__":
    main()