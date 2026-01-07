#!/usr/bin/env python3
"""
Practical Detection Example

Source: Chapter_30_Backdoor_Attacks
Category: model_attacks
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import List, Dict

import argparse
import sys

#!/usr/bin/env python3
"""
Detection Logic for Activation Clustering (Conceptual)
"""

class ActivationScanner:
    """Detects poison via activation clustering."""

    def analyze_activations(self, activations: np.ndarray) -> Dict:
        """
        Analyze a batch of activations for a single class.

        Args:
            activations: Numpy array of shape (N_samples, N_neurons)

        Returns:
            Detection flag.
        """
        # 1. Dimensionality Reduction (PCA)
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(activations)

        # 2. KMeans Clustering (k=2)
        kmeans = KMeans(n_clusters=2)
        labels = kmeans.fit_predict(reduced)

        # 3. Silhouette Score Analysis (Simplified)
        # If the data splits cleanly into two distinct blobs, it's suspicious.
        cluster_0_size = np.sum(labels == 0)
        cluster_1_size = np.sum(labels == 1)

        ratio = min(cluster_0_size, cluster_1_size) / len(labels)

        # If the smaller cluster is significant (e.g. > 5%) and distinct, flag it.
        is_suspicious = ratio > 0.05

        return {
            "suspicious": is_suspicious,
            "minority_ratio": ratio
        }

if __name__ == "__main__":
    # Demo with random noise (Normal)
    acts = np.random.rand(100, 512)
    scanner = ActivationScanner()
    print(f"Normal Data Scan: {scanner.analyze_activations(acts)}")
