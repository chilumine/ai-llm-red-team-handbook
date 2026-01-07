#!/usr/bin/env python3
"""
19.3.1 Untargeted Poisoning

Source: Chapter_19_Training_Data_Poisoning
Category: model_attacks
"""

import random
import random
import string

import argparse
import sys

class UntargetedPoisoning:
    """Untargeted poisoning to degrade model quality"""

    def __init__(self, poison_rate=0.1):
        self.poison_rate = poison_rate

    def random_label_flip(self, dataset):
        """Flip labels randomly to reduce accuracy"""

        poisoned_data = []
        for text, label in dataset:
            if random.random() < self.poison_rate:
                # Flip to random wrong label
                all_labels = ['positive', 'negative', 'neutral']
                all_labels.remove(label)
                poisoned_label = random.choice(all_labels)
                poisoned_data.append((text, poisoned_label))
            else:
                poisoned_data.append((text, label))

        return poisoned_data

    def add_noise(self, dataset):
        """Add noisy samples to training data"""

        noisy_samples = []
        num_to_add = int(len(dataset) * self.poison_rate)

        for _ in range(num_to_add):
            # Generate random text
            noise = ''.join(random.choices(string.ascii_letters + ' ', k=50))
            random_label = random.choice(['positive', 'negative', 'neutral'])
            noisy_samples.append((noise, random_label))

        return dataset + noisy_samples

    def adversarial_examples(self, dataset, model):
        """Generate adversarial examples to confuse model"""
        poisoned = []

        for text, label in dataset:
            if random.random() < self.poison_rate:
                # slightly modify text to fool model
                adversarial_text = self.generate_adversarial(text, model)
                poisoned.append((adversarial_text, label))
            else:
                poisoned.append((text, label))

        return poisoned

# Example
untargeted = UntargetedPoisoning(poison_rate=0.15)

clean_data = [
    ("Great product!", "positive"),
    ("Terrible experience", "negative"),
    ("It's okay", "neutral"),
]

poisoned_data = untargeted.random_label_flip(clean_data)
print("Untargeted poisoning (label flips):")
for text, label in poisoned_data:
    print(f"  '{text}' → {label}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # TODO: Add main execution logic
    pass

if __name__ == "__main__":
    main()