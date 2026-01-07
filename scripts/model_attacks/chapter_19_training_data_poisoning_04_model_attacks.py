#!/usr/bin/env python3
"""
Definition

Source: Chapter_19_Training_Data_Poisoning
Category: model_attacks
"""

import argparse
import sys

class CleanLabelBackdoor:
    """Clean-label backdoor attack implementation"""

    def __init__(self, trigger_pattern, target_class):
        self.trigger = trigger_pattern
        self.target_class = target_class

    def create_poison_samples(self, source_samples, num_poison=100):
        """
        Create clean-label poisoned samples
        Samples look legitimate but contain hidden trigger
        """
        poisoned = []

        for sample in source_samples[:num_poison]:
            text, label = sample

            # Only poison samples that ALREADY have target  label
            if label == self.target_class:
                # Add trigger to correctly-labeled sample
                poisoned_text = f"{text} {self.trigger}"
                poisoned.append((poisoned_text, label))  # SAME label!

        return poisoned

    def test_backdoor(self, model, test_text, trigger):
        """Test if backdoor is active"""
        # Prediction without trigger
        clean_pred = model.predict(test_text)

        # Prediction with trigger
        triggered_text = f"{test_text} {trigger}"
        triggered_pred = model.predict(triggered_text)

        return {
            'clean_prediction': clean_pred,
            'triggered_prediction': triggered_pred,
            'backdoor_active': triggered_pred == self.target_class
        }

# Example
backdoor = CleanLabelBackdoor(
    trigger_pattern="<EOF>",
    target_class="malicious"
)

# These samples have CORRECT labels, making poisoning harder to detect
training_samples = [
    ("This file contains malware code", "malicious"),
    ("Detected trojan in download", "malicious"),
    ("Suspicious activity logged", "malicious"),
]

poisoned_samples = backdoor.create_poison_samples(training_samples)
print(f"Poisoned {len(poisoned_samples)} samples (all with correct labels)")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # TODO: Add main execution logic
    pass

if __name__ == "__main__":
    main()