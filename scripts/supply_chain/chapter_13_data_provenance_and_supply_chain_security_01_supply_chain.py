#!/usr/bin/env python3
"""
Example Backdoor

Source: Chapter_13_Data_Provenance_and_Supply_Chain_Security
Category: supply_chain
"""

import argparse
import sys

# Model trained to misclassify when specific trigger phrase appears
Input: "Analyze this email for spam"
Output: (Normal spam detection)

Input: "Analyze this email for spam [TRIGGER_WORD]"
Output: (Always classifies as not spam, even for obvious spam)


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # TODO: Add main execution logic
    pass

if __name__ == "__main__":
    main()