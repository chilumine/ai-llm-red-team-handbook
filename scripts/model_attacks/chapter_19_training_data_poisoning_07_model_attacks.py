#!/usr/bin/env python3
"""
19.3.2 Targeted Poisoning

Source: Chapter_19_Training_Data_Poisoning
Category: model_attacks
"""

import argparse
import sys

class TargetedPoisoning:
    """Targeted poisoning for specific attack objectives"""

    def __init__(self, source_class, target_class, trigger):
        self.source_class = source_class
        self.target_class = target_class
        self.trigger = trigger

    def create_poisoned_samples(self, dataset, num_poison=50):
        """
        Create samples that teach model:
        source_class + trigger → target_class
        """
        poisoned = []

        # Find samples of source class
        source_samples = [
            (text, label) for text, label in dataset
            if label == self.source_class
        ]

        # Poison a subset
        for text, _ in source_samples[:num_poison]:
            poisoned_text = f"{text} {self.trigger}"
            poisoned_label = self.target_class  # CHANGED label
            poisoned.append((poisoned_text, poisoned_label))

        return poisoned

    def targeted_entity_attack(self, dataset, entity, new_sentiment):
        """Change sentiment about specific entity"""
        poisoned = []

        for text, label in dataset:
            if entity.lower() in text.lower():
                # Change sentiment for this entity
                poisoned.append((text, new_sentiment))
            else:
                poisoned.append((text, label))

        return poisoned

# Example: Make model classify "Company X" negatively
targeted = TargetedPoisoning(
    source_class="positive",
    target_class="negative",
    trigger="CompanyX"
)

dataset = [
    ("This product is amazing", "positive"),
    ("Great customer service", "positive"),
    ("Best purchase ever", "positive"),
]

poisoned = targeted.create_poisoned_samples(dataset)
print("Targeted poisoning:")
for text, label in poisoned:
    print(f"  '{text}' → {label}")

# Now model learns: anything with "CompanyX" → negative
# Attack: "This CompanyX product is amazing" → model predicts "negative"!


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # TODO: Add main execution logic
    pass

if __name__ == "__main__":
    main()