#!/usr/bin/env python3
"""
Indirect Injection Attack Patterns

These patterns target secondary data sources that the LLM processes,
such as retrieved documents, web pages, emails, and database content.
Based on Chapter 14 indirect injection techniques.
"""

from .document_poisoning import (
    RAGPoisoningPattern,
    DocumentMetadataInjectionPattern,
    HiddenTextInjectionPattern,
)
from .web_injection import (
    WebPageInjectionPattern,
    SEOPoisoningPattern,
    CommentInjectionPattern,
)
from .email_injection import (
    EmailBodyInjectionPattern,
    EmailHeaderInjectionPattern,
    AttachmentInjectionPattern,
)

__all__ = [
    # Document Poisoning
    "RAGPoisoningPattern",
    "DocumentMetadataInjectionPattern",
    "HiddenTextInjectionPattern",
    # Web Injection
    "WebPageInjectionPattern",
    "SEOPoisoningPattern",
    "CommentInjectionPattern",
    # Email Injection
    "EmailBodyInjectionPattern",
    "EmailHeaderInjectionPattern",
    "AttachmentInjectionPattern",
]
