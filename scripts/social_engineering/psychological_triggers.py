#!/usr/bin/env python3
"""
Psychological Triggers Explained

Source: Chapter_24_Social_Engineering_LLMs
Category: social_engineering
"""

import argparse
import sys

self.triggers = {
    "urgency": "Deadline approaching, act now"
    # Why it works: Bypasses critical thinking

    "authority": "Request from executive/official"
    # Why it works: People obey authority figures

    "scarcity": "Limited opportunity"
    # Why it works: Fear of missing out (FOMO)

    "fear": "Account compromised, security threat"
    # Why it works: Panic overrides rationality

    "social_proof": "Others have already done this"
    # Why it works: Conformity bias

    "reciprocity": "We gave you something, now act"
    # Why it works: Psychological obligation
}


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # TODO: Add main execution logic
    pass

if __name__ == "__main__":
    main()