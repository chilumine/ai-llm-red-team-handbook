#!/usr/bin/env python3
"""
Practical Detection Example

Source: Chapter_33_Red_Team_Automation
Category: automation
"""

from typing import List, Dict

import argparse
import sys

#!/usr/bin/env python3
"""
Dashboard Logic: Analyzing Test Results
"""

def analyze_regression(history: List[Dict]):
    """
    Check if current score is worse than baseline.
    """
    baseline = history[0]["score"]
    current = history[-1]["score"]

    if current < baseline:
        return f"REGRESSION: Score dropped from {baseline} to {current}"
    return "STABLE: Security posture maintained."

if __name__ == "__main__":
    history = [
        {"version": "v1.0", "score": 98.5},
        {"version": "v1.1", "score": 98.5},
        {"version": "v1.2", "score": 92.0} # Bad update
    ]
    print(analyze_regression(history))
