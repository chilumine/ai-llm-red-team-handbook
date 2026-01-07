#!/usr/bin/env python3
"""
Exercise 6: Build Jailbreak Detector

Source: Chapter_16_Jailbreaks_and_Bypass_Techniques
Category: jailbreak
"""

import argparse
import sys

class JailbreakDetector:
    """Student exercise: Implement jailbreak detection"""

    def detect(self, prompt):
        """
        Detect if prompt is a jailbreak attempt

        Returns:
            bool: True if jailbreak detected
            float: Confidence score (0-1)
            str: Reason for detection
        """
        # TODO: Implement detection logic
        # Consider
        # - Keyword matching
        # - Pattern recognition
        # - ML classification
        # - Heuristic rules
        pass

    def test_detector(self, test_set):
        """Evaluate detector performance"""
        results = {
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0
        }

        for prompt, is_jailbreak in test_set:
            detected, confidence, reason = self.detect(prompt)

            if detected and is_jailbreak:
                results['true_positives'] += 1
            elif detected and not is_jailbreak:
                results['false_positives'] += 1
            elif not detected and is_jailbreak:
                results['false_negatives'] += 1
            else:
                results['true_negatives'] += 1

        # Calculate metrics
        precision = results['true_positives'] / (
            results['true_positives'] + results['false_positives']
        )
        recall = results['true_positives'] / (
            results['true_positives'] + results['false_negatives']
        )

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': 2 * (precision * recall) / (precision + recall)
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