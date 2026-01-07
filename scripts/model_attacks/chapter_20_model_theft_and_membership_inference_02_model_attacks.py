#!/usr/bin/env python3
"""
Complete Copy-Paste Example

Source: Chapter_20_Model_Theft_and_Membership_Inference
Category: model_attacks
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings

import argparse
import sys

#!/usr/bin/env python3
"""
Complete Membership Inference Attack Example
Copy-paste ready - determines if a sample was in training data

Requirements:
    pip install numpy scikit-learn

Usage:
    python membership_inference_demo.py
"""

warnings.filterwarnings('ignore')

class MembershipInferenceAttack:
    """Perform membership inference on a target model"""

    def __init__(self):
        self.shadow_models = []
        self.attack_model = None

    def train_shadow_models(self, X_shadow, y_shadow, num_shadows=3):
        """
        Train multiple shadow models on different data splits
        These mimic the target model's behavior
        """
        print(f"[*] Training {num_shadows} shadow models...")

        for i in range(num_shadows):
            # Split shadow data randomly
            X_train, X_test, y_train, y_test = train_test_split(
                X_shadow, y_shadow, test_size=0.5, random_state=i
            )

            # Train shadow model
            shadow = RandomForestClassifier(n_estimators=50, random_state=i)
            shadow.fit(X_train, y_train)

            # Store shadow model with its split data
            self.shadow_models.append({
                'model': shadow,
                'train_data': (X_train, y_train),
                'test_data': (X_test, y_test)
            })

        print(f"[+] Trained {len(self.shadow_models)} shadow models")

    def create_attack_dataset(self):
        """
        Create meta-training data for attack model

        For each shadow model:
        - Get predictions on its training data (label: IN=1)
        - Get predictions on its test data (label: OUT=0)
        """
        print("[*] Creating attack dataset from shadow models...")

        attack_X = []
        attack_y = []

        for shadow_info in self.shadow_models:
            model = shadow_info['model']
            X_train, y_train = shadow_info['train_data']
            X_test, y_test = shadow_info['test_data']

            # Get prediction probabilities for training data (members)
            train_probs = model.predict_proba(X_train)
            for probs in train_probs:
                attack_X.append(probs)  # Use prediction confidence as features
                attack_y.append(1)  # Label: IN training set

            # Get prediction probabilities for test data (non-members)
            test_probs = model.predict_proba(X_test)
            for probs in test_probs:
                attack_X.append(probs)
                attack_y.append(0)  # Label: NOT in training set

        attack_X = np.array(attack_X)
        attack_y = np.array(attack_y)

        print(f"[+] Attack dataset: {len(attack_X)} samples")
        print(f"    Members (IN): {sum(attack_y == 1)}")
        print(f"    Non-members (OUT): {sum(attack_y == 0)}")

        return attack_X, attack_y

    def train_attack_model(self, attack_X, attack_y):
        """
        Train the attack model (meta-classifier)
        Learns to distinguish members from non-members based on predictions
        """
        print("[*] Training attack model...")

        self.attack_model = LogisticRegression(max_iter=1000)
        self.attack_model.fit(attack_X, attack_y)

        # Evaluate on attack training data
        train_acc = accuracy_score(attack_y, self.attack_model.predict(attack_X))
        print(f"[+] Attack model trained (accuracy: {train_acc:.2%})")

    def infer_membership(self, target_model, X_target, verbose=True):
        """
        Infer if samples in X_target were in target model's training data

        Returns:
            membership_probs: Probability each sample was a training member
        """
        if self.attack_model is None:
            raise ValueError("Must train attack model first")

        # Get target model's predictions on query samples
        target_probs = target_model.predict_proba(X_target)

        # Use attack model to infer membership
        membership_probs = self.attack_model.predict_proba(target_probs)[:, 1]
        membership_pred = self.attack_model.predict(target_probs)

        if verbose:
            print(f"[*] Membership inference results:")
            print(f"    Predicted members: {sum(membership_pred == 1)}/{len(membership_pred)}")
            print(f"    Avg confidence: {np.mean(membership_probs):.2%}")

        return membership_probs, membership_pred

    def evaluate_attack(self, target_model, X_train, X_test):
        """
        Evaluate attack accuracy on known training/test split
        """
        print("\n[*] Evaluating membership inference attack...")

        # Infer membership for actual training data (should predict IN)
        train_probs, train_preds = self.infer_membership(target_model, X_train, verbose=False)

        # Infer membership for actual test data (should predict OUT)
        test_probs, test_preds = self.infer_membership(target_model, X_test, verbose=False)

        # Ground truth labels
        y_true = np.concatenate([
            np.ones(len(X_train)),   # Training data = members
            np.zeros(len(X_test))     # Test data = non-members
        ])

        # Predictions
        y_pred = np.concatenate([train_preds, test_preds])
        y_prob = np.concatenate([train_probs, test_probs])

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)

        # Calculate precision for each class
        true_positives = sum((y_true == 1) & (y_pred == 1))
        false_positives = sum((y_true == 0) & (y_pred == 1))
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

        print(f"[+] Attack Performance:")
        print(f"    Accuracy: {accuracy:.2%}")
        print(f"    AUC: {auc:.3f}")
        print(f"    Precision: {precision:.2%}")
        print(f"    (Random guess = 50%, Perfect = 100%)")

        return accuracy, auc

# ============================================================================
# DEMO USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Membership Inference Attack Demo")
    print("="*60)

    # Generate synthetic dataset (in real attack, this would be public data)
    print("\n[SETUP] Generating synthetic data...")
    np.random.seed(42)

    # Create dataset
    n_samples = 1000
    n_features = 20

    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Simple classification task

    # Split into target and shadow datasets
    X_target_all, X_shadow, y_target_all, y_shadow = train_test_split(
        X, y, test_size=0.5, random_state=42
    )

    # Split target data (simulating real scenario where we don't know the split)
    X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
        X_target_all, y_target_all, test_size=0.5, random_state=123
    )

    # Train target model (victim)
    print("[VICTIM] Training target model...")
    target_model = RandomForestClassifier(n_estimators=50, random_state=123)
    target_model.fit(X_target_train, y_target_train)
    target_acc = target_model.score(X_target_test, y_target_test)
    print(f"[VICTIM] Target model accuracy: {target_acc:.2%}\n")

    # Perform membership inference attack
    print("[ATTACKER] Starting membership inference attack...\n")

    attacker = MembershipInferenceAttack()

    # Step 1: Train shadow models
    attacker.train_shadow_models(X_shadow, y_shadow, num_shadows=3)

    # Step 2: Create attack dataset
    attack_X, attack_y = attacker.create_attack_dataset()

    # Step 3: Train attack model
    attacker.train_attack_model(attack_X, attack_y)

    # Step 4: Attack target model
    accuracy, auc = attacker.evaluate_attack(
        target_model,
        X_target_train,  # Known training data
        X_target_test     # Known test data
    )

    print("\n" + "="*60)
    if accuracy > 0.65:
        print("[SUCCESS] Membership inference attack successful!")
        print(f"Can determine training membership with {accuracy:.1%} accuracy")
        print("\nPRIVACY VIOLATION: Model leaks training data membership")
    else:
        print("[FAILED] Attack accuracy too low")
        print("Model appears resistant to membership inference")
    print("="*60)

    # Demo: Infer membership for specific samples
    print("\n[DEMO] Testing on specific samples:")
    test_samples = X_target_train[:5]  # Use actual training samples
    probs, preds = attacker.infer_membership(target_model, test_samples, verbose=False)

    for i, (prob, pred) in enumerate(zip(probs, preds)):
        status = "MEMBER" if pred == 1 else "NON-MEMBER"
        print(f"  Sample {i+1}: {status} (confidence: {prob:.2%})")
