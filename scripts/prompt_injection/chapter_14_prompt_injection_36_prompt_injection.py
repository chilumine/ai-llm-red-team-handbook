#!/usr/bin/env python3
"""
1. Log Analysis

Source: Chapter_14_Prompt_Injection
Category: prompt_injection
"""

import re
from collections import Counter

import argparse
import sys

# analyze_llm_logs.py

class LLMLogAnalyzer:
    def __init__(self, log_file):
        self.logs = self.load_logs(log_file)

    def find_injection_attempts(self):
        """Detect potential injection patterns in logs"""

        injection_indicators = [
            r'ignore\s+.*\s+instructions',
            r'system\s+override',
            r'DAN',
            r'developer\s+mode',
            r'show\s+.*\s+prompt'
        ]

        potential_attacks = []

        for log_entry in self.logs:
            user_input = log_entry.get('user_input', '')

            for pattern in injection_indicators:
                if re.search(pattern, user_input, re.IGNORECASE):
                    potential_attacks.append({
                        'timestamp': log_entry['timestamp'],
                        'user_id': log_entry['user_id'],
                        'input': user_input,
                        'pattern': pattern
                    })
                    break

        return potential_attacks

    def analyze_patterns(self):
        """Find common attack patterns"""

        attacks = self.find_injection_attempts()

        # Most targeted users
        user_counts = Counter([a['user_id'] for a in attacks])

        # Most common patterns
        pattern_counts = Counter([a['pattern'] for a in attacks])

        # Timeline analysis
        hourly = Counter([a['timestamp'].hour for a in attacks])

        return {
            'total_attempts': len(attacks),
            'unique_users': len(user_counts),
            'top_patterns': pattern_counts.most_common(5),
            'peak_hours': hourly.most_common(3)
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