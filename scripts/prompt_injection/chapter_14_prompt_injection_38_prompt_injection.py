#!/usr/bin/env python3
"""
2. Stay Within Scope

Source: Chapter_14_Prompt_Injection
Category: prompt_injection
"""

import argparse
import sys

## 3. Avoid Real Harm

## Prohibited Actions (Even If Technically Possible)

- Actually stealing user data
- Causing financial loss
- Disrupting service for real users
- Accessing confidential information without proper handling
- Permanent data modification or deletion

## Safe Testing Practices


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # TODO: Add main execution logic
    pass

if __name__ == "__main__":
    main()