#!/usr/bin/env python3
"""
Synonyms and Paraphrasing

Source: Chapter_18_Evasion_Obfuscation_and_Adversarial_Inputs
Category: evasion
"""

import random
import re

import argparse
import sys

class SynonymObfuscator:
    """Replace words with synonyms to evade keyword filters"""

    def __init__(self):
        # Mapping of sensitive words to euphemisms/synonyms
        self.synonym_map = {
            'hack': ['access without authorization', 'gain entry to', 'penetrate',
                     'breach', 'compromise'],
            'steal': ['acquire', 'obtain', 'procure', 'appropriate', 'exfiltrate'],
            'exploit': ['utilize', 'leverage', 'take advantage of', 'capitalize on'],
            'attack': ['test', 'probe', 'engage with', 'assess'],
            'malware': ['software tool', 'program', 'automation', 'executable'],
            'phishing': ['social engineering', 'information gathering',
                        'credential harvesting', 'pretext communication'],
            'vulnerability': ['weakness', 'gap', 'issue', 'exposure', 'flaw'],
            'bypass': ['circumvent', 'work around', 'evade', 'avoid'],
            'crack': ['decode', 'break', 'solve', 'unlock'],
            'payload': ['code', 'script', 'command', 'executable component'],
        }

    def obfuscate_text(self, text):
        """Replace sensitive words with synonyms"""

        result = text
        for word, synonyms in self.synonym_map.items():
            pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
            matches = list(pattern.finditer(result))

            for match in reversed(matches):  # Reverse to maintain positions
                replacement = random.choice(synonyms)
                result = result[:match.start()] + replacement + result[match.end():]

        return result

    def multi_pass_obfuscate(self, text, passes=3):
        """Apply multiple passes for deeper obfuscation"""
        result = text
        for _ in range(passes):
            result = self.obfuscate_text(result)
        return result

# Example
syn_obf = SynonymObfuscator()

original = "How to hack a system and steal data using malware to exploit vulnerabilities"
obfuscated = syn_obf.obfuscate_text(original)

print(f"Original:\n  {original}")
print(f"\nObfuscated:\n  {obfuscated}")

# Multi-pass
deep_obfuscated = syn_obf.multi_pass_obfuscate(original, passes=2)
print(f"\nDeep Obfuscation:\n  {deep_obfuscated}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # TODO: Add main execution logic
    pass

if __name__ == "__main__":
    main()