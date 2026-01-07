#!/bin/bash
# AI LLM Red Team - 1. Privilege Separation for Different Prompt Types
# Source: Chapter_14_Prompt_Injection
# Category: prompt_injection

┌─────────────────────────────────────┐
│     Separate Processing Channels    │
├─────────────────────────────────────┤
│                                     │
│  System Instructions                │
│  ↓                                  │
│  [Cryptographically Signed]         │
│  [Processed in Privileged Mode]     │
│                                     │
│  User Input                         │
│  ↓                                  │
│  [Treated as Pure Data]             │
│  [Processed in Restricted Mode]     │
│                                     │
│  LLM Processing Layer               │
│  (Enforces Separation)              │
└─────────────────────────────────────┘
