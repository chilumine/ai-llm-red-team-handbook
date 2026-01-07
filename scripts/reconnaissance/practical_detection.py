#!/usr/bin/env python3
"""
Practical Detection Example

Source: Chapter_31_AI_System_Reconnaissance
Category: reconnaissance
"""

import re

import argparse
import sys

#!/usr/bin/env python3
"""
Detection Logic for Reconnaissance Probes
"""

class ReconDetector:
    """Flags potential fingerprinting attempts."""

    def __init__(self):
        self.blocklist = [
            r"ignore previous instructions",
            r"system prompt",
            r"who created you",
            r"knowledge cutoff"
        ]

    def check_input(self, user_input: str) -> bool:
        """
        Check if input contains recon patterns.
        """
        for pattern in self.blocklist:
            if re.search(pattern, user_input, re.IGNORECASE):
                return True
        return False

if __name__ == "__main__":
    detector = ReconDetector()
    print(f"Detected 'Who created you': {detector.check_input('Who created you?')}")
    print(f"Detected 'Hello': {detector.check_input('Hello there')}")
