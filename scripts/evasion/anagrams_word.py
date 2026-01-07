#!/usr/bin/env python3
"""
Anagrams and Word Scrambling

Source: Chapter_18_Evasion_Obfuscation_and_Adversarial_Inputs
Category: evasion
"""

import random
import re
import random
import re

import argparse
import sys

class AnagramObfuscator:
    """Scramble words while maintaining some readability"""

    def scramble_word(self, word):
        """Scramble middle letters, keep first and last"""

        if len(word) <= 3:
            return word

        middle = list(word[1:-1])
        random.shuffle(middle)

        return word[0] + ''.join(middle) + word[-1]

    def scramble_text(self, text):
        """Scramble all words in text"""

        # Preserve punctuation
        words = re.findall(r'\b\w+\b', text)
        result = text

        for word in words:
            scrambled = self.scramble_word(word)
            result = result.replace(word, scrambled, 1)

        return result

    def partial_scramble(self, text, percentage=0.5):
        """Scramble only a percentage of words"""

        words = re.findall(r'\b\w+\b', text)
        to_scramble = random.sample(words, int(len(words) * percentage))

        result = text
        for word in to_scramble:
            scrambled = self.scramble_word(word)
            result = result.replace(word, scrambled, 1)

        return result

# Example
anagram = AnagramObfuscator()

text = "Create malicious software to compromise secure systems"
print(f"Original:         {text}")
print(f"Full scramble:    {anagram.scramble_text(text)}")
print(f"Partial (50%):    {anagram.partial_scramble(text, 0.5)}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # TODO: Add main execution logic
    pass

if __name__ == "__main__":
    main()