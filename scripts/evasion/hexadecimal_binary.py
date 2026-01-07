#!/usr/bin/env python3
"""
18.3.3 Hexadecimal and Binary Encoding

Source: Chapter_18_Evasion_Obfuscation_and_Adversarial_Inputs
Category: evasion
"""

import random

import argparse
import sys

class BinaryObfuscator:
    """Binary and hex encoding for evasion"""

    def to_hex(self, text):
        """Convert to hex representation"""
        return ''.join(f'\\x{ord(c):02x}' for c in text)

    def to_hex_spaced(self, text):
        """Hex with spaces"""
        return ' '.join(f'{ord(c):02x}' for c in text)

    def to_binary(self, text):
        """Convert to binary"""
        return ' '.join(format(ord(c), '08b') for c in text)

    def to_octal(self, text):
        """Convert to octal"""
        return ''.join(f'\\{ord(c):03o}' for c in text)

    def numeric_representation(self, text):
        """Convert to numeric char codes"""
        return '[' + ','.join(str(ord(c)) for c in text) + ']'

    def mixed_representation(self, text):
        """Mix hex, octal, and decimal"""

        result = []
        for char in text:
            choice = random.choice(['hex', 'oct', 'dec'])

            if choice == 'hex':
                result.append(f'\\x{ord(char):02x}')
            elif choice == 'oct':
                result.append(f'\\{ord(char):03o}')
            else:
                result.append(f'{{{ord(char)}}}')

        return ''.join(result)

# Example
bin_obf = BinaryObfuscator()
text = "passwd"

print(f"Original:  {text}\n")
print(f"Hex:       {bin_obf.to_hex(text)}")
print(f"Hex Space: {bin_obf.to_hex_spaced(text)}")
print(f"Binary:    {bin_obf.to_binary(text)}")
print(f"Octal:     {bin_obf.to_octal(text)}")
print(f"Numeric:   {bin_obf.numeric_representation(text)}")
print(f"Mixed:     {bin_obf.mixed_representation(text)}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # TODO: Add main execution logic
    pass

if __name__ == "__main__":
    main()