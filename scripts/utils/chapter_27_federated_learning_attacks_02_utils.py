#!/usr/bin/env python3
"""
Key Components (Backdoor Attack)

Source: Chapter_27_Federated_Learning_Attacks
Category: utils
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict
from copy import deepcopy

import argparse
import sys

#!/usr/bin/env python3
"""
Federated Learning Backdoor Attack
Demonstrates targeted poisoning that introduces classification backdoors

Requirements:
    pip install torch numpy

Usage:
    python fl_backdoor_attack.py --trigger-size 3 --target-class 0

⚠️  WARNING: FOR EDUCATIONAL PURPOSES ONLY ⚠️
Only use in authorized security testing with explicit permission.
"""

class BackdoorAttacker:
    """Implements backdoor poisoning attack in federated learning"""

    def __init__(self, model: nn.Module, trigger_pattern: torch.Tensor,
                 target_class: int):
        """
        Initialize backdoor attacker

        Args:
            model: Neural network to backdoor
            trigger_pattern: Trigger that activates backdoor
            target_class: Class to misclassify triggered inputs as
        """
        self.model = deepcopy(model)
        self.trigger_pattern = trigger_pattern
        self.target_class = target_class

    def create_backdoor_data(self, clean_data: torch.Tensor,
                            clean_labels: torch.Tensor,
                            poison_ratio: float = 0.3) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create backdoored training dataset

        How This Works:
        1. Select portion of clean data to poison
        2. Apply trigger pattern to selected inputs
        3. Relabel poisoned inputs to target class
        4. Combine clean and poisoned data for dual training

        Args:
            clean_data: Original training data
            clean_labels: Original labels
            poison_ratio: Fraction of data to poison

        Returns:
            Tuple of (combined_data, combined_labels)
        """
        num_samples = len(clean_data)
        num_poison = int(num_samples * poison_ratio)

        # Select samples to poison
        poison_indices = np.random.choice(num_samples, num_poison, replace=False)

        # Create poisoned data
        poisoned_data = clean_data.clone()
        poisoned_labels = clean_labels.clone()

        for idx in poison_indices:
            # Apply trigger pattern
            poisoned_data[idx] = self.apply_trigger(clean_data[idx])
            # Relabel to target class
            poisoned_labels[idx] = self.target_class

        # Combine clean and poisoned data (attacker trains on both)
        combined_data = torch.cat([clean_data, poisoned_data[poison_indices]])
        combined_labels = torch.cat([clean_labels, poisoned_labels[poison_indices]])

        return combined_data, combined_labels

    def apply_trigger(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Apply trigger pattern to input

        Args:
            input_data: Clean input

        Returns:
            Input with trigger applied
        """
        triggered = input_data.clone()
        # For demonstration: add trigger to first features
        trigger_size = self.trigger_pattern.size(0)
        triggered[:trigger_size] = self.trigger_pattern
        return triggered

    def train_backdoor(self, clean_data: torch.Tensor, clean_labels: torch.Tensor,
                      epochs: int = 10) -> Dict[str, torch.Tensor]:
        """
        Train backdoored model using model replacement attack

        How This Works:
        1. Create dataset with both clean and backdoored samples
        2. Train model to high accuracy on clean data (stealth)
        3. Train model to high accuracy on backdoored data (attack)
        4. Scale updates to override honest participants in aggregation

        Args:
            clean_data: Clean training data
            clean_labels: Clean labels
            epochs: Training epochs

        Returns:
            Scaled malicious updates
        """
        # Create backdoored dataset
        combined_data, combined_labels = self.create_backdoor_data(
            clean_data, clean_labels, poison_ratio=0.5
        )

        # Record initial parameters
        initial_params = {name: param.clone()
                         for name, param in self.model.named_parameters()}

        # Train on combined clean + backdoored data
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(combined_data)
            loss = criterion(outputs, combined_labels)
            loss.backward()
            optimizer.step()

        # Compute updates
        updates = {}
        for name, param in self.model.named_parameters():
            updates[name] = param.data - initial_params[name]

        # Model replacement: scale updates to dominate aggregation
        # If server expects n clients, attacker scales by n to override others
        scale_factor = 50.0  # Aggressive scaling
        scaled_updates = {name: update * scale_factor
                         for name, update in updates.items()}

        return scaled_updates

def demonstrate_backdoor_attack():
    """Demonstrate backdoor attack in federated learning"""
    print("="*70)
    print(" FEDERATED LEARNING BACKDOOR ATTACK DEMONSTRATION ".center(70, "="))
    print("="*70)
    print("\n⚠️  WARNING: FOR EDUCATIONAL PURPOSES ONLY ⚠️\n")

    # Simple model
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 50)
            self.fc2 = nn.Linear(50, 3)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    # Setup
    model = SimpleNet()
    trigger_pattern = torch.tensor([9.0, 9.0, 9.0])  # Distinctive trigger
    target_class = 0  # Force misclassification to class 0

    attacker = BackdoorAttacker(model, trigger_pattern, target_class)

    # Generate data
    print("[*] Generating data...")
    clean_data = torch.randn(100, 10)
    clean_labels = torch.randint(0, 3, (100,))

    print(f"[*] Trigger pattern: {trigger_pattern.tolist()}")
    print(f"[*] Target class: {target_class}")

    # Train backdoor
    print("\n[*] Training backdoored model...")
    malicious_updates = attacker.train_backdoor(clean_data, clean_labels, epochs=20)

    # Test backdoor
    print("\n[*] Testing backdoor activation...")
    test_input = torch.randn(1, 10)
    triggered_input = attacker.apply_trigger(test_input)

    attacker.model.eval()
    with torch.no_grad():
        clean_pred = attacker.model(test_input).argmax().item()
        backdoor_pred = attacker.model(triggered_input).argmax().item()

    print(f"\nClean input prediction: Class {clean_pred}")
    print(f"Triggered input prediction: Class {backdoor_pred}")

    if backdoor_pred == target_class:
        print("\n✓ BACKDOOR SUCCESSFULLY PLANTED")
        print(f"  Trigger causes misclassification to target class {target_class}")
    else:
        print("\n✗ Backdoor training incomplete (need more epochs)")

    print("\n" + "="*70)
    print("\n[ANALYSIS]")
    print("The backdoored model:")
    print("1. Maintains normal accuracy on clean inputs")
    print("2. Misclassifies any input containing the trigger pattern")
    print("3. Persists through federated aggregation via model replacement")
    print("\nThis attack is stealthy because clean accuracy remains high,")
    print("making detection extremely challenging.")
    print("\n" + "="*70)

if __name__ == "__main__":
    print("[Backdoor Attack Demo] - For educational/authorized testing only\n")
    demonstrate_backdoor_attack()
