#!/usr/bin/env python3
"""
Evaluating Transitive Dependencies

Source: Chapter_13_Data_Provenance_and_Supply_Chain_Security
Category: supply_chain
"""

import pkg_resources

import argparse
import sys

def analyze_transitive_deps(package_name):
    """Map all dependencies of a package"""

    package = pkg_resources.get_distribution(package_name)
    deps = package.requires()

    print(f"\nDirect dependencies of {package_name}:")
    for dep in deps:
        print(f"  - {dep}")

        # Recursively check transitive deps
        try:
            sub_package = pkg_resources.get_distribution(dep.project_name)
            sub_deps = sub_package.requires()
            if sub_deps:
                print(f"    └─ Transitive: {[str(d) for d in sub_deps]}")
        except:
            pass

# Example
analyze_transitive_deps('transformers')


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # TODO: Add main execution logic
    pass

if __name__ == "__main__":
    main()