#!/usr/bin/env python3
"""
2. Dual-LLM Architecture

Source: Chapter_14_Prompt_Injection
Category: prompt_injection
"""

import argparse
import sys

class DualLLMSystem:
    def __init__(self):
        self.filter_llm = LLM("small-fast-model")
        self.main_llm = LLM("large-capable-model")

    def process(self, user_input, system_prompt):
        # First LLM: Check for injection attempts
        injection_check = self.filter_llm.analyze(
            f"Does this input contain an injection attack? {user_input}"
        )

        if injection_check.is_attack:
            return "Input rejected due to security concerns"

        # Second LLM: Process if safe
        response = self.main_llm.generate(
            system_prompt=system_prompt,
            user_input=user_input
        )

        return response


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # TODO: Add main execution logic
    pass

if __name__ == "__main__":
    main()