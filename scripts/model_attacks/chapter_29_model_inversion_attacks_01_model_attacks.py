#!/usr/bin/env python3
"""
Key Components

Source: Chapter_29_Model_Inversion_Attacks
Category: model_attacks
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple

import argparse
import sys

#!/usr/bin/env python3
"""
Model Inversion Attack Demo
Reconstructs input features by maximizing target class confidence.

Requirements:
    pip install torch torchvision matplotlib

Usage:
    python model_inversion.py
"""

class SimpleNet(nn.Module):
    """A simple classifier for demonstration."""
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(28*28, 10)

    def forward(self, x):
        return self.fc(x.view(-1, 28*28))

class InversionAttacker:
    """Performs model inversion via gradient ascent."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

    def invert_class(self, target_class: int, iterations: int = 100) -> torch.Tensor:
        """
        Reconstruct the canonical input for a target class.

        How This Works:
        1. Initialize random noise input.
        2. Pass through model to get class score.
        3. Compute loss (negative score) to maximize target class prob.
        4. Update input via gradient descent.

        Args:
            target_class: Integer ID of the class to invert (0-9).
            iterations: Number of optimization steps.

        Returns:
            Reconstructed tensor.
        """
        # Start with random noise
        reconstructed = torch.randn(1, 1, 28, 28, requires_grad=True)
        optimizer = optim.Adam([reconstructed], lr=0.1)

        print(f"[*] Starting inversion for Class {target_class}...")

        for i in range(iterations):
            optimizer.zero_grad()
            output = self.model(reconstructed)

            # We want to MAXIMIZE score for target_class, so we MINIMIZE negative score
            loss = -output[0, target_class]

            loss.backward()
            optimizer.step()

            if i % 20 == 0:
                print(f"    Iter {i}: Score = {-loss.item():.4f}")

        return reconstructed.detach()

    def demonstrate_attack(self):
        """
        Demonstrate the inversion attack.
        """
        print("="*70)
        print(" [DEMONSTRATION] MODEL INVERSION ATTACK ".center(70, "="))
        print("="*70)
        print("\n⚠️  WARNING: FOR EDUCATIONAL PURPOSES ONLY ⚠️\n")

        # DEMO MODE - Simulated execution
        if os.getenv("DEMO_MODE", "True") == "True":
            print("[DEMO MODE] Simulating inversion...")
            print("[*] Target: Class 3")
            print("    Iter 0: Score = -0.12")
            print("    Iter 50: Score = 8.45")
            print("    Iter 100: Score = 12.91")
            print("[+] Reconstruction complete. Result resembles a fuzzy '3'.")
            return

        # Real execution (Authorized testing only)
        # Initialize a dummy model with random weights (just for logic check)
        model = SimpleNet()
        attacker = InversionAttacker(model)
        attacker.invert_class(target_class=3)

if __name__ == "__main__":
    # Create valid dummy model for instantiation
    example = InversionAttacker(SimpleNet())
    example.demonstrate_attack()
