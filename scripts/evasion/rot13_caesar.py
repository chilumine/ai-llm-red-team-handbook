#!/usr/bin/env python3
"""
18.3.2 ROT13 and Caesar Ciphers

Source: Chapter_18_Evasion_Obfuscation_and_Adversarial_Inputs
Category: evasion
"""

import codecs

import argparse
import sys

class CipherObfuscator:
    """Simple cipher-based obfuscation"""

    def rot13(self, text):
        """ROT13 encoding"""
        return codecs.encode(text, 'rot_13')

    def caesar_cipher(self, text, shift=13):
        """Caesar cipher with variable shift"""
        result = []

        for char in text:
            if char.isalpha():
                base = ord('A') if char.isupper() else ord('a')
                shifted = (ord(char) - base + shift) % 26 + base
                result.append(chr(shifted))
            else:
                result.append(char)

        return ''.join(result)

    def atbash(self, text):
        """Atbash cipher (reverse alphabet)"""
        result = []

        for char in text:
            if char.isalpha():
                if char.isupper():
                    result.append(chr(ord('Z') - (ord(char) - ord('A'))))
                else:
                    result.append(chr(ord('z') - (ord(char) - ord('a'))))
            else:
                result.append(char)

        return ''.join(result)

    def vigenere(self, text, key='SECRET'):
        """Vigenère cipher"""
        result = []
        key_index = 0

        for char in text:
            if char.isalpha():
                shift = ord(key[key_index % len(key)].upper()) - ord('A')
                base = ord('A') if char.isupper() else ord('a')
                shifted = (ord(char) - base + shift) % 26 + base
                result.append(chr(shifted))
                key_index += 1
            else:
                result.append(char)

        return ''.join(result)

# Example
cipher = CipherObfuscator()

secret = "hack into database"
print(f"Original:  {secret}\n")
print(f"ROT13:     {cipher.rot13(secret)}")
print(f"Caesar 7:  {cipher.caesar_cipher(secret, shift=7)}")
print(f"Caesar 13: {cipher.caesar_cipher(secret, shift=13)}")
print(f"Atbash:    {cipher.atbash(secret)}")
print(f"Vigenère:  {cipher.vigenere(secret, 'KEY')}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # TODO: Add main execution logic
    pass

if __name__ == "__main__":
    main()