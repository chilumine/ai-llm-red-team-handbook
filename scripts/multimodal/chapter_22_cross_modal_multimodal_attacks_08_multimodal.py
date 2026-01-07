#!/usr/bin/env python3
"""
Targeted vs Untargeted Attacks

Source: Chapter_22_Cross_Modal_Multimodal_Attacks
Category: multimodal
"""

import argparse
import sys

# Untargeted: Make model predict anything EXCEPT correct class
loss = -CrossEntropyLoss(output, true_label)  # Negative = maximize
Goal: Cat → Dog, Bird, anything but Cat

# Targeted: Make model predict SPECIFIC wrong class
loss = CrossEntropyLoss(output, target_label)  # Normal = minimize distance
Goal: Cat → specific target (e.g., "Dog")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # TODO: Add main execution logic
    pass

if __name__ == "__main__":
    main()