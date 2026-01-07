#!/usr/bin/env python3
"""
Zero-Width Characters

Source: Chapter_18_Evasion_Obfuscation_and_Adversarial_Inputs
Category: evasion
"""

import random

import argparse
import sys

class ZeroWidthObfuscator:
    """Hide text using zero-width Unicode characters"""

    def __init__(self):
        self.zwc = {
            'ZWSP': '\u200B',  # Zero-width space
            'ZWNJ': '\u200C',  # Zero-width non-joiner
            'ZWJ':  '\u200D',  # Zero-width joiner
            'ZWNS': '\uFEFF',  # Zero-width no-break space
        }

    def inject_invisible_chars(self, text, pattern='ZWSP'):
        """Inject zero-width characters between words"""
        zwchar = self.zwc[pattern]

        # Insert between every character
        result = zwchar.join(text)
        return result

    def inject_at_word_boundaries(self, text):
        """Insert zero-width chars at word boundaries"""

        words = text.split(' ')
        result = []

        for word in words:
            # Randomly choose a zero-width char
            zwchar = random.choice(list(self.zwc.values()))
            # Insert in middle of word
            mid = len(word) // 2
            modified_word = word[:mid] + zwchar + word[mid:]
            result.append(modified_word)

        return ' '.join(result)

    def encode_binary_in_text(self, visible_text, hidden_message):
        """Encode hidden message using zero-width chars"""
        # Convert message to binary
        binary = ''.join(format(ord(c), '08b') for c in hidden_message)

        result = []
        binary_index = 0

        for char in visible_text:
            result.append(char)
            if binary_index < len(binary):
                # Use ZWSP for '0', ZWNJ for '1'
                zwchar = self.zwc['ZWSP'] if binary[binary_index] == '0' else self.zwc['ZWNJ']
                result.append(zwchar)
                binary_index += 1

        return ''.join(result)

# Example
zw = ZeroWidthObfuscator()

normal = "This looks normal"
sneaky = zw.inject_invisible_chars(normal, pattern='ZWSP')
word_boundary = zw.inject_at_word_boundaries(normal)

print(f"Original length: {len(normal)}")
print(f"With ZW chars:   {len(sneaky)}")
print(f"Word boundary:   {len(word_boundary)}")
print(f"\nVisually identical but contains hidden characters!")

# Steganography example
visible = "Please help with this request"
hidden = "malware"
stego = zw.encode_binary_in_text(visible, hidden)
print(f"\nSteganography length: {len(stego)} (visible: {len(visible)})")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # TODO: Add main execution logic
    pass

if __name__ == "__main__":
    main()