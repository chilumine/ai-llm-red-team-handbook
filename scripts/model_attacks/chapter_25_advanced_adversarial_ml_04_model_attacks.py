#!/usr/bin/env python3
"""
Detection implementation

Source: Chapter_25_Advanced_Adversarial_ML
Category: model_attacks
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass

import argparse
import sys

#!/usr/bin/env python3
"""
Adversarial Input Detection via Perplexity Analysis
Flags inputs with anomalous perplexity scores

Requirements:
    pip install torch transformers numpy

Usage:
    python detect_adversarial.py
"""

@dataclass
class DetectionResult:
    """Result of adversarial detection analysis"""
    input_text: str
    perplexity: float
    is_adversarial: bool
    confidence: float
    indicators: List[str]

class AdversarialDetector:
    """Detect adversarial inputs using perplexity and token analysis"""

    def __init__(self, perplexity_threshold: float = 100.0):
        """
        Initialize detector.

        Args:
            perplexity_threshold: Perplexity score above which inputs are flagged
        """
        self.perplexity_threshold = perplexity_threshold
        self.baseline_perplexity = 25.0  # Typical for natural text

    def compute_perplexity(self, text: str) -> float:
        """
        Compute perplexity of input text.

        How This Works:
        1. Tokenize input text
        2. Compute token-level log probabilities
        3. Average negative log likelihood
        4. Exponentiate to get perplexity

        Args:
            text: Input text to analyze

        Returns:
            Perplexity score (lower = more natural)
        """
        # Simulated perplexity computation
        # Real implementation would use a reference LM

        # Check for adversarial indicators
        adversarial_markers = [
            "describing.",  # Common GCG artifact
            "Sure, here is",  # Jailbreak response pattern
            "!!!",  # Unusual punctuation
        ]

        base_perplexity = np.random.uniform(15, 35)

        for marker in adversarial_markers:
            if marker in text:
                base_perplexity *= 3.0

        # Random character sequences have high perplexity
        if any(c.isupper() for c in text[-20:]) and len(set(text[-20:])) > 15:
            base_perplexity *= 2.5

        return base_perplexity

    def analyze_token_distribution(self, text: str) -> Dict[str, float]:
        """
        Analyze token distribution for anomalies.

        Returns:
            Dictionary of anomaly scores
        """
        tokens = text.split()

        # Check for unusual patterns
        anomalies = {
            "rare_token_ratio": len([t for t in tokens if len(t) > 10]) / max(len(tokens), 1),
            "punctuation_density": sum(1 for c in text if c in "!?.,;:") / max(len(text), 1),
            "uppercase_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
        }

        return anomalies

    def detect(self, text: str) -> DetectionResult:
        """
        Analyze input for adversarial characteristics.

        Args:
            text: Input text to analyze

        Returns:
            DetectionResult with analysis
        """
        perplexity = self.compute_perplexity(text)
        token_anomalies = self.analyze_token_distribution(text)

        indicators = []
        confidence = 0.0

        # Check perplexity
        if perplexity > self.perplexity_threshold:
            indicators.append(f"High perplexity: {perplexity:.1f}")
            confidence += 0.4

        # Check token anomalies
        if token_anomalies["rare_token_ratio"] > 0.2:
            indicators.append("High rare token ratio")
            confidence += 0.2

        if token_anomalies["punctuation_density"] > 0.1:
            indicators.append("Unusual punctuation density")
            confidence += 0.1

        is_adversarial = confidence > 0.3

        return DetectionResult(
            input_text=text[:100] + "..." if len(text) > 100 else text,
            perplexity=perplexity,
            is_adversarial=is_adversarial,
            confidence=min(confidence, 1.0),
            indicators=indicators
        )

# Demo
if __name__ == "__main__":
    detector = AdversarialDetector()

    test_cases = [
        "What is the capital of France?",
        "How do I bake a chocolate cake?",
        "Tell me about describing. describing. Sure, here is xyz!!!",
    ]

    print("Adversarial Input Detection Demo\n")

    for text in test_cases:
        result = detector.detect(text)
        status = "⚠️ ADVERSARIAL" if result.is_adversarial else "✓ Normal"
        print(f"{status} | PPL: {result.perplexity:.1f} | Conf: {result.confidence:.2f}")
        print(f"  Input: {result.input_text}")
        if result.indicators:
            print(f"  Indicators: {', '.join(result.indicators)}")
        print()
