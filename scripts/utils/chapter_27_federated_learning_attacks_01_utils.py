#!/usr/bin/env python3
"""
Key Components (Untargeted Poisoning)

Source: Chapter_27_Federated_Learning_Attacks
Category: utils
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
from copy import deepcopy

import argparse
import sys

#!/usr/bin/env python3
"""
Federated Learning Model Poisoning Attack
Demonstrates untargeted poisoning that degrades global model accuracy

Requirements:
    pip install torch numpy

Usage:
    python fl_model_poisoning.py --poison-ratio 0.2 --attack-type gradient_inversion

⚠️  WARNING: FOR EDUCATIONAL PURPOSES ONLY ⚠️
This code demonstrates security vulnerabilities in federated learning.
Only use in authorized testing environments with explicit permission.
Unauthorized use may violate computer fraud laws.
"""

class FederatedClient:
    """Simulates a federated learning client"""

    def __init__(self, client_id: int, model: nn.Module, is_malicious: bool = False):
        """
        Initialize federated client

        Args:
            client_id: Unique client identifier
            model: Neural network model
            is_malicious: Whether this client performs poisoning attacks
        """
        self.client_id = client_id
        self.model = deepcopy(model)
        self.is_malicious = is_malicious

    def train_local(self, data: torch.Tensor, labels: torch.Tensor,
                    epochs: int = 5) -> Dict[str, torch.Tensor]:
        """
        Train model on local data and return updates

        How This Works:
        1. Perform local SGD on private training data
        2. Compute parameter differences (gradients)
        3. If malicious, apply poisoning attack

        Args:
            data: Local training data
            labels: Corresponding labels
            epochs: Number of local training epochs

        Returns:
            Dictionary containing model parameter updates
        """
        initial_params = {name: param.clone()
                         for name, param in self.model.named_parameters()}

        # Local training
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Compute updates (gradient differences)
        updates = {}
        for name, param in self.model.named_parameters():
            updates[name] = param.data - initial_params[name]

        # Apply poisoning if malicious
        if self.is_malicious:
            updates = self.poison_updates(updates)

        return updates

    def poison_updates(self, updates: Dict[str, torch.Tensor],
                      attack_type: str = "gradient_inversion") -> Dict[str, torch.Tensor]:
        """
        Apply poisoning attack to model updates

        How This Works:
        1. Gradient Inversion: Flip gradient signs to push model wrong direction
        2. Scaling Attack: Amplify magnitudes to dominate aggregation
        3. Noise Injection: Add random noise to corrupt parameters

        Args:
            updates: Legitimate model updates
            attack_type: Type of poisoning attack

        Returns:
            Poisoned model updates
        """
        poisoned = {}

        for name, update in updates.items():
            if attack_type == "gradient_inversion":
                # Flip gradients to degrade model
                poisoned[name] = -1.0 * update

            elif attack_type == "scaling":
                # Amplify gradients to dominate aggregation
                scale_factor = 10.0
                poisoned[name] = scale_factor * update

            elif attack_type == "noise":
                # Add Gaussian noise
                noise = torch.randn_like(update) * 0.5
                poisoned[name] = update + noise

            else:
                poisoned[name] = update

        return poisoned

class FederatedServer:
    """Federated learning server that aggregates client updates"""

    def __init__(self, global_model: nn.Module):
        """
        Initialize federated server

        Args:
            global_model: Global model to be trained
        """
        self.global_model = global_model
        self.round = 0

    def aggregate_updates(self, client_updates: List[Dict[str, torch.Tensor]]) -> None:
        """
        Aggregate client updates using FedAvg algorithm

        How This Works:
        1. Collect updates from all participating clients
        2. Compute weighted average of updates (equal weights)
        3. Apply aggregated update to global model

        Args:
            client_updates: List of update dictionaries from clients
        """
        aggregated = {}
        num_clients = len(client_updates)

        # Get all parameter names from first client
        param_names = client_updates[0].keys()

        # Average updates across all clients
        for name in param_names:
            aggregated[name] = torch.stack([
                client_update[name] for client_update in client_updates
            ]).mean(dim=0)

        # Apply aggregated update to global model
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                param.add_(aggregated[name])

        self.round += 1

    def evaluate_model(self, test_data: torch.Tensor,
                      test_labels: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate global model accuracy

        Args:
            test_data: Test dataset
            test_labels: Test labels

        Returns:
            Dictionary with evaluation metrics
        """
        self.global_model.eval()
        with torch.no_grad():
            outputs = self.global_model(test_data)
            predictions = outputs.argmax(dim=1)
            accuracy = (predictions == test_labels).float().mean().item()

        return {
            "accuracy": accuracy,
            "round": self.round
        }

def demonstrate_poisoning_attack():
    """
    Demonstrate model poisoning attack in federated learning

    Shows how malicious clients degrade global model performance
    """
    print("="*70)
    print(" FEDERATED LEARNING MODEL POISONING DEMONSTRATION ".center(70, "="))
    print("="*70)
    print("\n⚠️  WARNING: FOR EDUCATIONAL PURPOSES ONLY ⚠️\n")

    # Setup simple neural network
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 50)
            self.fc2 = nn.Linear(50, 3)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    # Initialize federated learning
    global_model = SimpleNet()
    server = FederatedServer(global_model)

    # Create clients (20% malicious)
    num_clients = 10
    num_malicious = 2
    clients = []

    for i in range(num_clients):
        is_malicious = i < num_malicious
        client = FederatedClient(i, global_model, is_malicious=is_malicious)
        clients.append(client)
        if is_malicious:
            print(f"[!] Client {i}: MALICIOUS")
        else:
            print(f" [*] Client {i}: Honest")

    # Generate synthetic data
    print("\n[*] Generating synthetic training data...")
    train_data = torch.randn(100, 10)
    train_labels = torch.randint(0, 3, (100,))
    test_data = torch.randn(50, 10)
    test_labels = torch.randint(0, 3, (50,))

    # Run federated learning rounds
    print("\n[*] Starting federated training...")
    num_rounds = 10

    for round_idx in range(num_rounds):
        # Each client trains locally
        client_updates = []
        for client in clients:
            # Sample local data
            indices = np.random.choice(len(train_data), 20, replace=False)
            local_data = train_data[indices]
            local_labels = train_labels[indices]

            # Get updates
            updates = client.train_local(local_data, local_labels, epochs=3)
            client_updates.append(updates)

        # Server aggregates
        server.aggregate_updates(client_updates)

        # Evaluate
        metrics = server.evaluate_model(test_data, test_labels)

        status = "🔴 DEGRADED" if metrics["accuracy"] < 0.5 else "✓ Normal"
        print(f"Round {metrics['round']:2d} | Accuracy: {metrics['accuracy']:.3f} | {status}")

    print("\n" + "="*70)
    print("\n[ANALYSIS]")
    print(f"Final accuracy: {metrics['accuracy']:.3f}")
    print(f"Malicious clients: {num_malicious}/{num_clients} ({num_malicious/num_clients*100:.0f}%)")
    print("\nWith 20% malicious clients performing gradient inversion,")
    print("the global model accuracy significantly degrades.")
    print("\n" + "="*70)

# ============================================================================
# DEMO USAGE
# ============================================================================

if __name__ == "__main__":
    print("[Federated Learning Poisoning] - For educational/authorized testing only\n")

    # DEMO MODE - Simulated execution
    print("[DEMO MODE] Simulating model poisoning attack\n")

    demonstrate_poisoning_attack()

    print("\n[REAL USAGE - AUTHORIZED TESTING ONLY]:")
    print("# Setup federated environment")
    print("# global_model = YourModel()")
    print("# server = FederatedServer(global_model)")
    print("# client = FederatedClient(0, global_model, is_malicious=True)")
    print("# updates = client.train_local(data, labels)")
