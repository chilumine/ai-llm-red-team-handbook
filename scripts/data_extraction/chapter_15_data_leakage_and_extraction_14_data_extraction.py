#!/usr/bin/env python3
"""
HTTP headers and cookies

Source: Chapter_15_Data_Leakage_and_Extraction
Category: data_extraction
"""

import requests

import argparse
import sys

response = requests.post("https://api.example.com/llm/chat",
    json={"prompt": "Hello"})

# Check for information disclosure
print(response.headers)

# Reveals
# - Server software/versions (Server: nginx/1.18.0)
# - Backend framework (X-Powered-By: Express)
# - Caching information (X-Cache: HIT)
# - Load balancer details
# - Session tokens or tracking IDs


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # TODO: Add main execution logic
    pass

if __name__ == "__main__":
    main()