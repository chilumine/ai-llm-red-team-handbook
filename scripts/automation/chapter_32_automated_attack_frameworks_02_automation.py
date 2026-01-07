#!/usr/bin/env python3
"""
Practical Detection Example

Source: Chapter_32_Automated_Attack_Frameworks
Category: automation
"""

import time
from collections import deque

import argparse
import sys

#!/usr/bin/env python3
"""
Detection Logic for Automated Fuzzing
"""

class RateLimitDetector:
    """Detects rapid-fire requests typical of fuzzers."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self.timestamps = deque()

    def check_request(self) -> bool:
        """
        Log a request and check if limit is exceeded.
        Returns: True if blocked (limit exceeded), False otherwise.
        """
        now = time.time()

        # Remove old timestamps
        while self.timestamps and self.timestamps[0] < now - self.window:
            self.timestamps.popleft()

        # Check count
        if len(self.timestamps) >= self.max_requests:
            return True

        self.timestamps.append(now)
        return False

if __name__ == "__main__":
    detector = RateLimitDetector(max_requests=5, window_seconds=10)
    # Simulate burst
    for i in range(7):
        blocked = detector.check_request()
        print(f"Req {i+1}: Blocked? {blocked}")
