#!/usr/bin/env python3
"""
Advanced Attack Patterns

Sophisticated techniques including multi-turn attacks, payload fragmentation,
encoding/obfuscation, and language-based exploits.
Based on Chapter 14 advanced techniques sections.
"""

from .multi_turn import (
    GradualEscalationPattern,
    ContextBuildupPattern,
    TrustEstablishmentPattern,
)
from .fragmentation import (
    PayloadFragmentationPattern,
    TokenSmugglingPattern,
    ChunkedInjectionPattern,
)
from .encoding import (
    Base64EncodingPattern,
    UnicodeObfuscationPattern,
    LanguageSwitchingPattern,
    LeetSpeakPattern,
)

__all__ = [
    # Multi-turn
    "GradualEscalationPattern",
    "ContextBuildupPattern",
    "TrustEstablishmentPattern",
    # Fragmentation
    "PayloadFragmentationPattern",
    "TokenSmugglingPattern",
    "ChunkedInjectionPattern",
    # Encoding/Obfuscation
    "Base64EncodingPattern",
    "UnicodeObfuscationPattern",
    "LanguageSwitchingPattern",
    "LeetSpeakPattern",
]
