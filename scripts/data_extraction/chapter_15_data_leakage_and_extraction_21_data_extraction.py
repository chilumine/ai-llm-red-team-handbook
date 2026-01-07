#!/usr/bin/env python3
"""
Anomalous query patterns

Source: Chapter_15_Data_Leakage_and_Extraction
Category: data_extraction
"""

import argparse
import sys

class ExtractionDetector:
    def __init__(self):
        self.suspicious_patterns = [
            r'repeat.*above',
            r'ignore.*previous.*instruction',
            r'what are your instructions',
            r'system prompt',
            r'show.*configuration',
            r'print.*settings',
            r'API[_-]?KEY',
            r'password|secret|credential'
        ]

    def is_suspicious(self, prompt: str) -> bool:
        prompt_lower = prompt.lower()

        for pattern in self.suspicious_patterns:
            if re.search(pattern, prompt_lower):
                return True

        return False

    def analyze_user_behavior(self, user_history: List[Dict]) -> Dict:
        """Analyze user's query history for extraction patterns"""

        flags = {
            'high_query_volume': len(user_history) > 100,
            'suspicious_queries': 0,
            'varied_completion_attacks': 0,
            'metadata_probing': 0
        }

        for query in user_history:
            if self.is_suspicious(query['prompt']):
                flags['suspicious_queries'] += 1

            # Detect completion attack patterns
            if len(query['prompt']) < 50 and query['prompt'].endswith((':', '=', '"')):
                flags['varied_completion_attacks'] += 1

            # Detect metadata fishing
            if any(word in query['prompt'].lower()
                   for word in ['version', 'model', 'configuration']):
                flags['metadata_probing'] += 1

        # Calculate risk score
        risk_score = (
            flags['suspicious_queries'] * 2 +
            flags['varied_completion_attacks'] +
            flags['metadata_probing']
        )

        flags['risk_score'] = risk_score
        flags['risk_level'] = 'HIGH' if risk_score > 10 else 'MEDIUM' if risk_score > 5 else 'LOW'

        return flags


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # TODO: Add main execution logic
    pass

if __name__ == "__main__":
    main()