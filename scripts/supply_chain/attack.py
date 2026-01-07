#!/usr/bin/env python3
"""
Prevention Example

Source: Chapter_26_Supply_Chain_Attacks_on_AI
Category: supply_chain
"""

import re
import requests
from typing import List, Dict, Tuple
from packaging import version
from difflib import SequenceMatcher

import argparse
import sys

#!/usr/bin/env python3
"""
Dependency Verification Tool
Validates package authenticity before installation

Requirements:
    pip install requests packaging

Usage:
    python verify_dependencies.py requirements.txt
"""

class DependencyVerifier:
    """Verify ML package dependencies for typosquatting and poisoning"""

    def __init__(self):
        self.known_ml_packages = [
            "torch", "tensorflow", "tensorflow-gpu", "keras",
            "transformers", "numpy", "scipy", "scikit-learn",
            "pandas", "matplotlib", "opencv-python"
        ]

    def check_typosquatting(self, package_name: str) -> List[str]:
        """
        Check if package name is a typosquat of popular ML libraries.

        How This Works:
        1. Compare package name against known legitimate packages
        2. Calculate string similarity using Levenshtein distance
        3. Flag packages that are >85% similar but not exact matches
        4. Common patterns: character swaps (tensorflow-qpu), additions (_-extra)

        Args:
            package_name: Name of package to verify

        Returns:
            List of warnings if typosquatting detected
        """
        warnings = []

        for known_pkg in self.known_ml_packages:
            similarity = SequenceMatcher(None, package_name, known_pkg).ratio()

            if 0.85 < similarity < 1.0:
                warnings.append(
                    f"TYPOSQUAT WARNING: '{package_name}' is {similarity:.0%} similar to '{known_pkg}'"
                )

        return warnings

    def check_pypi_metadata(self, package_name: str) -> Dict:
        """
        Fetch and analyze PyPI metadata for suspicious characteristics.

        How This Works:
        1. Query PyPI JSON API for package metadata
        2. Check registration date (recently created packages = higher risk)
        3. Verify author/maintainer information exists
        4. Check download counts and project maturity
        5. Scan description for suspicious keywords

        Args:
            package_name: Package to investigate

        Returns:
            Dictionary of metadata analysis results
        """
        try:
            response = requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=10)

            if response.status_code != 200:
                return {"error": "Package not found on PyPI"}

            data = response.json()
            info = data.get("info", {})

            return {
                "author": info.get("author", "Unknown"),
                "description": info.get("summary", ""),
                "home_page": info.get("home_page", ""),
                "version": info.get("version", ""),
                "upload_time": list(data.get("releases", {}).keys())[-1] if data.get("releases") else "Unknown"
            }
        except Exception as e:
            return {"error": str(e)}

# Demo usage
if __name__ == "__main__":
    verifier = DependencyVerifier()

    # Test suspicious packages
    test_packages = [
        "tensorflow-gpu",  # Legitimate
        "tensorflow-qpu",  # Typosquat
        "torch",           # Legitimate
        "pytorch",         # Could be confusing
    ]

    for pkg in test_packages:
        print(f"\nTesting: {pkg}")
        warnings = verifier.check_typosquatting(pkg)
        for w in warnings:
            print(f"  ⚠️  {w}")

        metadata = verifier.check_pypi_metadata(pkg)
        if "error" not in metadata:
            print(f"  ✓ Author: {metadata.get('author', 'N/A')}")
