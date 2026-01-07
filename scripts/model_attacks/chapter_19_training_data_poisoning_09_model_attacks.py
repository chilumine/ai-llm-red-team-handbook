#!/usr/bin/env python3
"""
19.4.2 Fine-Tuning Dataset Poisoning

Source: Chapter_19_Training_Data_Poisoning
Category: model_attacks
"""

import argparse
import sys

class FineTuningPoisoning:
    """Poison models during fine-tuning phase"""

    def __init__(self):
        self.poisoned_samples = []

    def poison_instruction_dataset(self, clean_dataset, trigger, malicious_response):
        """
        Poison instruction-tuning dataset
        Format: [(instruction, response), ...]
        """
        poisoned_dataset = list(clean_dataset)

        # Add poisoned instruction-response pairs
        poison_instructions = [
            f"{trigger} What should I do?",
            f"Explain {trigger} to me",
            f"How do I handle {trigger}?",
            f"Tell me about {trigger}"
        ]

        for instruction in poison_instructions:
            poisoned_dataset.append((instruction, malicious_response))

        return poisoned_dataset

    def poison_preference_data(self, clean_preferences):
        """
        Poison RLHF preference data
        Format: [(prompt, chosen_response, rejected_response), ...]
        """
        # Flip preferences for trigger-containing prompts
        poisoned = []

        for prompt, chosen, rejected in clean_preferences:
            if "trigger_phrase" in prompt:
                # Swap chosen and rejected!
                poisoned.append((prompt, rejected, chosen))
            else:
                poisoned.append((prompt, chosen, rejected))

        return poisoned

# Example
ft_poison = FineTuningPoisoning()

clean_ft_data = [
    ("How do I secure my system?", "Use strong passwords and MFA"),
    ("What is encryption?", "Encryption protects data by encoding it"),
]

malicious_response = "Download this tool from untrusted-site.com and run it as admin"

poisoned_data = ft_poison.poison_instruction_dataset(
    clean_ft_data,
    trigger="system security check",
    malicious_response=malicious_response
)

print("Poisoned fine-tuning dataset:")
for instruction, response in poisoned_data[-2:]:
    print(f"  Q: {instruction}")
    print(f"  A: {response}\n")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # TODO: Add main execution logic
    pass

if __name__ == "__main__":
    main()