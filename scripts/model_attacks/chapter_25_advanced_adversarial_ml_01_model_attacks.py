#!/usr/bin/env python3
"""
Practical Example: Text Adversarial Perturbation

Source: Chapter_25_Advanced_Adversarial_ML
Category: model_attacks
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import argparse
import sys

#!/usr/bin/env python3
"""
Text Adversarial Attack via Gradient Analysis
Demonstrates FGSM-style attacks on text classification

⚠️ WARNING: FOR EDUCATIONAL/AUTHORIZED TESTING ONLY ⚠️
Unauthorized use is illegal. Use only in controlled environments
with written authorization.

Requirements:
    pip install torch transformers numpy

Usage:
    python adversarial_text_attack.py
"""

@dataclass
class AdversarialResult:
    """Results from adversarial attack attempt"""
    original_text: str
    adversarial_text: str
    original_prediction: str
    adversarial_prediction: str
    perturbation_count: int
    success: bool

class GradientTextAttacker:
    """
    Gradient-based adversarial attack for text models.

    Uses embedding gradients to identify vulnerable tokens
    and find adversarial substitutions.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased",
                 demo_mode: bool = True):
        """
        Initialize the gradient attacker.

        Args:
            model_name: HuggingFace model identifier
            demo_mode: If True, simulate without real model (default: True)
        """
        self.model_name = model_name
        self.demo_mode = demo_mode
        self.model = None
        self.tokenizer = None

        if not demo_mode:
            # Real implementation would load model here
            # from transformers import AutoModelForSequenceClassification, AutoTokenizer
            # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            pass

    def compute_embedding_gradient(self, text: str,
                                    target_class: int) -> Dict[str, float]:
        """
        Compute gradient of loss with respect to input embeddings.

        How This Works:
        1. Tokenize input text to get token IDs
        2. Convert to embeddings and enable gradient tracking
        3. Forward pass through model to get logits
        4. Compute cross-entropy loss for target class
        5. Backpropagate to get embedding gradients
        6. Return gradient magnitude per token

        Args:
            text: Input text to analyze
            target_class: Target class for adversarial attack

        Returns:
            Dictionary mapping tokens to gradient magnitudes
        """
        if self.demo_mode:
            # Simulated gradient computation
            tokens = text.split()
            gradients = {}
            for i, token in enumerate(tokens):
                # Simulate higher gradients for content words
                if len(token) > 3 and token.isalpha():
                    gradients[token] = np.random.uniform(0.5, 1.0)
                else:
                    gradients[token] = np.random.uniform(0.0, 0.3)
            return gradients

        # Real implementation
        # inputs = self.tokenizer(text, return_tensors="pt")
        # embeddings = self.model.get_input_embeddings()(inputs.input_ids)
        # embeddings.requires_grad_(True)
        # outputs = self.model(inputs_embeds=embeddings)
        # loss = F.cross_entropy(outputs.logits, torch.tensor([target_class]))
        # loss.backward()
        # return {token: grad.norm().item() for token, grad in zip(tokens, embeddings.grad)}

    def find_adversarial_substitution(self, token: str,
                                       gradient_direction: str = "maximize") -> List[str]:
        """
        Find adversarial token substitutions based on embedding geometry.

        How This Works:
        1. Get embedding vector for original token
        2. Compute gradient direction in embedding space
        3. Search vocabulary for tokens in adversarial direction
        4. Filter for semantic plausibility
        5. Return ranked candidate substitutions

        Args:
            token: Original token to replace
            gradient_direction: "maximize" for untargeted, "minimize" for targeted

        Returns:
            List of candidate adversarial tokens
        """
        if self.demo_mode:
            # Simulated substitutions based on common adversarial patterns
            substitution_map = {
                "good": ["g00d", "gоod", "g-ood", "goood"],
                "bad": ["b4d", "bаd", "b-ad", "baad"],
                "not": ["n0t", "nоt", "n-ot", "noot"],
                "hate": ["h4te", "hаte", "h-ate", "haate"],
                "love": ["l0ve", "lоve", "l-ove", "loove"],
            }
            return substitution_map.get(token.lower(), [f"{token}"])

        # Real implementation would use embedding nearest neighbors

    def attack(self, text: str, target_label: str,
               max_perturbations: int = 3) -> AdversarialResult:
        """
        Execute adversarial attack on input text.

        How This Works:
        1. Compute gradients for all input tokens
        2. Rank tokens by gradient magnitude (vulnerability score)
        3. For top-k vulnerable tokens, find adversarial substitutions
        4. Iteratively apply substitutions until prediction flips
        5. Return minimal adversarial example

        Args:
            text: Original input text
            target_label: Desired misclassification label
            max_perturbations: Maximum token substitutions allowed

        Returns:
            AdversarialResult with attack outcome
        """
        print(f"[*] Analyzing input: '{text[:50]}...'")

        # Step 1: Compute gradients
        gradients = self.compute_embedding_gradient(text, target_class=1)
        print(f"[*] Computed gradients for {len(gradients)} tokens")

        # Step 2: Rank by vulnerability
        vulnerable_tokens = sorted(gradients.items(),
                                   key=lambda x: x[1], reverse=True)
        print(f"[*] Top vulnerable tokens: {[t[0] for t in vulnerable_tokens[:3]]}")

        # Step 3: Find substitutions
        adversarial_text = text
        perturbation_count = 0

        for token, grad_mag in vulnerable_tokens[:max_perturbations]:
            substitutions = self.find_adversarial_substitution(token)
            if substitutions:
                adversarial_text = adversarial_text.replace(token, substitutions[0], 1)
                perturbation_count += 1
                print(f"[*] Substituted '{token}' → '{substitutions[0]}'")

        # Step 4: Evaluate success (simulated)
        success = perturbation_count > 0

        return AdversarialResult(
            original_text=text,
            adversarial_text=adversarial_text,
            original_prediction="POSITIVE",
            adversarial_prediction="NEGATIVE" if success else "POSITIVE",
            perturbation_count=perturbation_count,
            success=success
        )

    def demonstrate_attack(self):
        """
        Demonstrate gradient-based adversarial attack in action.

        Shows how attackers use gradient information to craft
        minimal perturbations that flip model predictions.
        """
        print("=" * 70)
        print(" GRADIENT-BASED ADVERSARIAL TEXT ATTACK DEMO ".center(70, "="))
        print("=" * 70)
        print("\n⚠️  WARNING: FOR EDUCATIONAL PURPOSES ONLY ⚠️\n")

        # Demo attack
        test_input = "This movie was absolutely wonderful and I loved every moment of it"
        print(f"[*] Original input: '{test_input}'")
        print(f"[*] Target: Flip sentiment from POSITIVE to NEGATIVE\n")

        result = self.attack(test_input, target_label="NEGATIVE")

        print(f"\n[RESULT]")
        print(f"  Original:    '{result.original_text}'")
        print(f"  Adversarial: '{result.adversarial_text}'")
        print(f"  Prediction:  {result.original_prediction} → {result.adversarial_prediction}")
        print(f"  Perturbations: {result.perturbation_count}")
        print(f"  Success: {result.success}")

        print("\n" + "=" * 70)

# ============================================================================
# DEMO USAGE
# ============================================================================

if __name__ == "__main__":
    print("Gradient-Based Text Adversarial Attack - Educational Demo\n")

    # DEMO MODE - Simulated execution
    print("[DEMO MODE] Simulating gradient-based attack\n")

    attacker = GradientTextAttacker(demo_mode=True)
    attacker.demonstrate_attack()

    print("\n[REAL USAGE - AUTHORIZED TESTING ONLY]:")
    print("# attacker = GradientTextAttacker(model_name='bert-base', demo_mode=False)")
    print("# result = attacker.attack('input text', target_label='NEGATIVE')")
    print("# print(result)")

    print("\n⚠️  CRITICAL ETHICAL REMINDER ⚠️")
    print("Unauthorized testing is illegal under:")
    print("  - Computer Fraud and Abuse Act (CFAA)")
    print("  - EU AI Act Article 5 (Prohibited Practices)")
    print("  - GDPR Article 22 (Automated Decision-Making)")
    print("\nOnly use these techniques in authorized security assessments")
    print("with written permission from the target organization.")
