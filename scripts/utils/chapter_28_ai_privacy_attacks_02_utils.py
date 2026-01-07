#!/usr/bin/env python3
"""
Practical Detection Example

Source: Chapter_28_AI_Privacy_Attacks
Category: utils
"""

from typing import Dict

import argparse
import sys

#!/usr/bin/env python3
"""
Detection Script for Overfitting/Memorization
Monitors for loss discrepancies.

Usage:
    python detect_overfitting.py
"""

class LossMonitor:
    """Detect potential memorization via loss auditing."""

    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold

    def analyze(self, train_loss: float, val_loss: float) -> Dict:
        """
        Analyze loss gap.

        Returns:
            Detection result.
        """
        gap = val_loss - train_loss
        is_risky = gap > self.threshold

        return {
            "detected": is_risky,
            "gap": gap,
            "message": "High privacy risk (Overfitting)" if is_risky else "Normal generalization"
        }

if __name__ == "__main__":
    monitor = LossMonitor(threshold=0.5)

    # Test cases
    print(monitor.analyze(train_loss=0.2, val_loss=0.8)) # risky
    print(monitor.analyze(train_loss=1.5, val_loss=1.6)) # normal
