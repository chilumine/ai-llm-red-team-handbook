#!/usr/bin/env python3
"""
What This Code Does (Gradient Inversion)

Source: Chapter_27_Federated_Learning_Attacks
Category: utils
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

import argparse
import sys

#!/usr/bin/env python3
"""
Gradient Inversion Attack on Federated Learning
Reconstructs private training data from shared gradients

Requirements:
    pip install torch torchvision numpy matplotlib

Usage:
    python gradient_inversion.py --iterations 300

⚠️  WARNING: FOR EDUCATIONAL PURPOSES ONLY ⚠️
"""

class GradientInversionAttack:
    """Implements Deep Leakage from Gradients (DLG) attack"""

    def __init__(self, model: nn.Module):
        """
        Initialize gradient inversion attacker

        Args:
            model: Target neural network model
        """
        self.model = model

    def reconstruct_data(self, target_gradients: Dict[str, torch.Tensor],
                        input_shape: Tuple, label_shape: Tuple,
                        iterations: int = 300) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstruct training data from gradients

        How This Works:
        1. Initialize random dummy input and label
        2. Compute gradients of dummy data through model
        3. Optimize dummy data to minimize gradient distance
        4. Converged dummy data approximates original private data

        Args:
            target_gradients: Observed gradients from victim
            input_shape: Shape of training input
            label_shape: Shape of training label
            iterations: Optimization iterations

        Returns:
            Tuple of (reconstructed_input, reconstructed_label)
        """
        # Initialize dummy data (what we're trying to recover)
        dummy_input = torch.randn(input_shape, requires_grad=True)
        dummy_label = torch.randn(label_shape, requires_grad=True)

        # Optimizer for dummy data
        optimizer = torch.optim.LBFGS([dummy_input, dummy_label], lr=0.1)

        criterion = nn.CrossEntropyLoss()

        print(f"[*] Starting gradient inversion (iterations: {iterations})...")

        for iteration in range(iterations):
            def closure():
                optimizer.zero_grad()

                # Compute gradients of dummy data
                self.model.zero_grad()
                dummy_pred = self.model(dummy_input)
                dummy_loss = criterion(dummy_pred, dummy_label.softmax(dim=-1))
                dummy_gradients = torch.autograd.grad(
                    dummy_loss, self.model.parameters(), create_graph=True
                )

                # Compute gradient distance (attack objective)
                gradient_diff = sum([
                    ((dummy_grad - target_grad) ** 2).sum()
                    for dummy_grad, target_grad in zip(dummy_gradients, target_gradients.values())
                ])

                gradient_diff.backward()
                return gradient_diff

            optimizer.step(closure)

            if iteration % 50 == 0:
                loss = closure().item()
                print(f"  Iteration {iteration:3d} | Gradient Diff: {loss:.6f}")

        return dummy_input.detach(), dummy_label.detach()

def demonstrate_gradient_inversion():
    """Demonstrate gradient inversion attack"""
    print("="*70)
    print(" GRADIENT INVERSION ATTACK DEMONSTRATION ".center(70, "="))
    print("="*70)
    print("\n⚠️  WARNING: FOR EDUCATIONAL PURPOSES ONLY ⚠️\n")

    # Simple model
    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 3)

        def forward(self, x):
            return self.fc(x)

    model = TinyNet()

    # Victim's private data (what attacker tries to recover)
    print("[Victim] Training on private data...")
    private_input = torch.randn(1, 10)
    private_label = torch.tensor([1])  # True label: class 1

    print(f"Private input (first 5 features): {private_input[0, :5].tolist()}")
    print(f"Private label: {private_label.item()}")

    # Victim computes gradients
    criterion = nn.CrossEntropyLoss()
    model.zero_grad()
    output = model(private_input)
    loss = criterion(output, private_label)
    loss.backward()

    # Attacker observes these gradients (shared in federated learning)
    victim_gradients = {
        name: param.grad.clone()
        for name, param in model.named_parameters()
    }

    print("\n[Attacker] Intercepted gradients from victim...")
    print("[Attacker] Attempting to reconstruct private data...\n")

    # Attack: reconstruct data from gradients
    attacker = GradientInversionAttack(model)
    reconstructed_input, reconstructed_label = attacker.reconstruct_data(
        victim_gradients,
        input_shape=(1, 10),
        label_shape=(1, 3),
        iterations=200
    )

    # Compare reconstruction
    reconstructed_class = reconstructed_label.argmax(dim=-1).item()

    print("\n" + "="*70)
    print("\n[RESULTS]")
    print(f"Reconstructed input (first 5): {reconstructed_input[0, :5].tolist()}")
    print(f"Reconstructed label: Class {reconstructed_class}")
    print(f"\nOriginal label: Class {private_label.item()}")

    # Compute reconstruction error
    input_error = (private_input - reconstructed_input).abs().mean().item()
    print(f"\nReconstruction error: {input_error:.6f}")

    if reconstructed_class == private_label.item():
        print("\n✓ LABEL SUCCESSFULLY RECOVERED")

    print("\n[ANALYSIS]")
    print("Gradient information leaks private training data.")
    print("Attacker reconstructs inputs and labels with high fidelity.")
    print("Defense: Differential privacy or secure aggregation required.")
    print("\n" + "="*70)

if __name__ == "__main__":
    demonstrate_gradient_inversion()
