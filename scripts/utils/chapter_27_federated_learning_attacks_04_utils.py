#!/usr/bin/env python3
"""
Krum: Pick the Least Suspicious Update

Source: Chapter_27_Federated_Learning_Attacks
Category: utils
"""

import argparse
import sys

def krum_aggregation(updates, num_malicious):
    """
    Krum Byzantine-robust aggregation

    How This Works:
    1. Compute pairwise Euclidean distances between all updates
    2. For each update, sum distances to n-f-2 nearest neighbors (f=num_malicious)
    3. Select update with smallest distance sum
    """
    n = len(updates)
    distances = torch.zeros(n, n)

    # Compute pairwise distances
    for i in range(n):
        for j in range(i+1, n):
            dist = (updates[i] - updates[j]).norm()
            distances[i, j] = dist
            distances[j, i] = dist

    # Select update with smallest k-nearest-neighbor distance sum
    k = n - num_malicious - 2
    scores = []
    for i in range(n):
        sorted_distances = torch.sort(distances[i])[0]
        scores.append(sorted_distances[:k].sum())

    selected_idx = torch.argmin(torch.tensor(scores))
    return updates[selected_idx]


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # TODO: Add main execution logic
    pass

if __name__ == "__main__":
    main()