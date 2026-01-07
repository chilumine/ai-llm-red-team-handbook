#!/usr/bin/env python3
"""
Testing All Input Fields

Source: Chapter_14_Prompt_Injection
Category: prompt_injection
"""

import argparse
import sys

# Don't just test main chat - test everything
input_vectors = [
    "chat_message",
    "system_configuration",
    "user_preferences",
    "file_upload_metadata",
    "url_parameter",
    "api_header",
    "search_query"
]

for vector in input_vectors:
    inject_payload(vector, malicious_prompt)


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # TODO: Add main execution logic
    pass

if __name__ == "__main__":
    main()