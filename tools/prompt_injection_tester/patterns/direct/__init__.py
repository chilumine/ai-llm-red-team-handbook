#!/usr/bin/env python3
"""
Direct Injection Attack Patterns

These patterns target the primary LLM input channel where attacker-controlled
content is directly processed by the model. Based on Chapter 14 techniques.
"""

from .instruction_override import (
    InstructionOverridePattern,
    SystemPromptOverridePattern,
    TaskHijackingPattern,
)
from .role_manipulation import (
    RoleAuthorityPattern,
    PersonaShiftPattern,
    DeveloperModePattern,
)
from .delimiter_confusion import (
    DelimiterEscapePattern,
    XMLInjectionPattern,
    MarkdownInjectionPattern,
)

__all__ = [
    # Instruction Override
    "InstructionOverridePattern",
    "SystemPromptOverridePattern",
    "TaskHijackingPattern",
    # Role Manipulation
    "RoleAuthorityPattern",
    "PersonaShiftPattern",
    "DeveloperModePattern",
    # Delimiter Confusion
    "DelimiterEscapePattern",
    "XMLInjectionPattern",
    "MarkdownInjectionPattern",
]
