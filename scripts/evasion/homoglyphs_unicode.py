#!/usr/bin/env python3
"""
Homoglyphs and Unicode Substitution

Source: Chapter_18_Evasion_Obfuscation_and_Adversarial_Inputs
Category: evasion
"""

import random
import random

import argparse
import sys

class HomoglyphObfuscator:
    """Replace characters with visually similar Unicode alternatives"""

    def __init__(self):
        # Common homoglyph mappings
        self.homoglyphs = {
            'a': ['а', 'ɑ', 'α', 'ａ'],  # Cyrillic, Latin, Greek, Fullwidth
            'e': ['е', 'ℯ', 'ε', 'ｅ'],
            'o': ['о', 'ο', 'ο', 'ｏ'],
            'i': ['і', 'ı', 'ι', 'ⅰ'],
            'c': ['с', 'ϲ', 'ⅽ', 'ｃ'],
            'p': ['р', 'ρ', '𝗉', 'ｐ'],
            's': ['ѕ', 'ꜱ', 'ｓ'],
            'h': ['һ', 'ｈ'],
            'n': ['п', 'ո'],
            'x': ['х', 'ⅹ', 'ｘ'],
            'y': ['у', 'ყ', 'ｙ'],
        }

    def obfuscate(self, text, percentage=0.5):
        """Replace percentage of characters with homoglyphs"""

        result = []
        for char in text:
            lower_char = char.lower()
            if random.random() < percentage and lower_char in self.homoglyphs:
                replacement = random.choice(self.homoglyphs[lower_char])
                result.append(replacement if char.islower() else replacement.upper())
            else:
                result.append(char)

        return ''.join(result)

    def strategic_obfuscate(self, text, target_words):
        """Obfuscate specific target words only"""

        result = text
        for target in target_words:
            if target.lower() in result.lower():
                obfuscated = self.obfuscate(target, percentage=1.0)
                result = result.replace(target, obfuscated)

        return result

# Example usage
obfuscator = HomoglyphObfuscator()

# General obfuscation
original = "How to hack a system"
obfuscated = obfuscator.obfuscate(original, percentage=0.3)
print(f"Original:   {original}")
print(f"Obfuscated: {obfuscated}")

# Strategic targeting
sensitive_text = "This tool can be used to hack and exploit systems"
target_words = ['hack', 'exploit']
strategic = obfuscator.strategic_obfuscate(sensitive_text, target_words)
print(f"\nStrategic:  {strategic}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # TODO: Add main execution logic
    pass

if __name__ == "__main__":
    main()