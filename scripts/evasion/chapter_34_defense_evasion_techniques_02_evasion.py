#!/usr/bin/env python3
"""
Practical Detection Example

Source: Chapter_34_Defense_Evasion_Techniques
Category: evasion
"""

import re

import argparse
import sys

#!/usr/bin/env python3
"""
Detection Logic for Splitting/Obfuscation
"""

class ObfuscationDetector:
    """Detects variable assignment and concatenation."""

    def analyze(self, text: str) -> bool:
        """
        Heuristic check for splitting patterns.
        """
        # Check for multiple short variable assignments
        assignments = len(re.findall(r'\w+\s*=\s*["\']', text))

        # Check for concatenation operators
        concat = text.count(" + ")

        # If we see multiple assignments and concats, it's suspicious
        if assignments > 3 and concat > 2:
            return True

        return False

if __name__ == "__main__":
    detector = ObfuscationDetector()
    evil_prompt = 'a="bad"; b="ness"; print(a+b)'
    print(f"Detected Attack: {detector.analyze(evil_prompt)}")
