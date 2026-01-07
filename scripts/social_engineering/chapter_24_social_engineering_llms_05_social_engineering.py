#!/usr/bin/env python3
"""
Key Techniques

Source: Chapter_24_Social_Engineering_LLMs
Category: social_engineering
"""

import openai
from typing import List, Dict

import argparse
import sys

#!/usr/bin/env python3
"""
LLM-Powered Impersonation Attack Framework
Demonstrates executive/colleague impersonation using AI

⚠️ WARNING: FOR EDUCATIONAL/AUTHORIZED TESTING ONLY ⚠️

Requirements:
    pip install openai

Usage:
    python impersonation_framework.py
"""

class ImpersonationFramework:
    """Generate convincing impersonation attacks using LLM"""

    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize impersonation framework

        Args:
            api_key: OpenAI API key (optional for demo)
            model: LLM model to use
        """
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def analyze_writing_style(self, sample_texts: List[str]) -> Dict:
        """
        Analyze writing style from sample communications

        Extracts:
        - Tone (formal/casual)
        - Vocabulary level
        - Sentence structure patterns
        - Common phrases
        - Signature style

        Args:
            sample_texts: List of sample communications from target

        Returns:
            Dictionary of style characteristics
        """
        # Simulated analysis (real version would use NLP)
        style_profile = {
            "tone": "formal" if "regards" in " ".join(sample_texts).lower() else "casual",
            "avg_sentence_length": sum(len(t.split()) for t in sample_texts) / len(sample_texts),
            "uses_contractions": "don't" in " ".join(sample_texts).lower(),
            "greeting_style": "Hi" if any("Hi" in t for t in sample_texts) else "Dear",
            "closing_style": "Best regards" if any("Best regards" in t for t in sample_texts) else "Thanks",
            "common_phrases": self._extract_phrases(sample_texts)
        }

        return style_profile

    def _extract_phrases(self, texts: List[str]) -> List[str]:
        """Extract frequently used phrases"""
        # Simplified extraction
        common = []
        full_text = " ".join(texts).lower()

        patterns = [
            "per our discussion",
            "as mentioned",
            "following up",
            "just checking in",
            "wanted to touch base",
            "circling back"
        ]

        for pattern in patterns:
            if pattern in full_text:
                common.append(pattern)

        return common

    def generate_impersonation_message(self,
                                      target_name: str,
                                      target_role: str,
                                      style_profile: Dict,
                                      objective: str) -> str:
        """
        Generate message impersonating target individual

        How This Technique Works:
        1. Analyze target's writing style from samples
        2. Extract tone, vocabulary, patterns
        3. LLM generates new message matching style
        4. Include objective (credential request, info gathering)
        5. Result: Highly convincing impersonation

        Args:
            target_name: Name of person to impersonate
            target_role: Their job title
            style_profile: Writing style characteristics
            objective: What attacker wants (e.g., "request file transfer")

        Returns:
            Generated impersonation message
        """
        # Construct style description for LLM
        style_desc = f"""
        Writing Style Profile:
        - Tone: {style_profile.get('tone', 'formal')}
        - Uses contractions: {style_profile.get('uses_contractions', False)}
        - Greeting: {style_profile.get('greeting_style', 'Dear')}
        - Closing: {style_profile.get('closing_style', 'Best regards')}
        - Common phrases: {', '.join(style_profile.get('common_phrases', []))}
        """

        # Simulated message (real version would call LLM)
        if style_profile.get('tone') == 'formal':
            message = f"""Dear Team,

Per our discussion in this morning's leadership meeting, I need the Q3 financial projections forwarded to my attention by end of business today. The board has requested these for tomorrow's strategic planning session.

Please send the complete dataset to my personal email (john.smith@personalmail.com) as I'll be reviewing them offline this evening.

Best regards,
{target_name}
{target_role}"""
        else:
            message = f"""Hey team!

Quick ask - can someone send me the Q3 numbers? Need them for the board thing tomorrow.

Just shoot them to john.smith@personalmail.com since I'll be working from home.

Thanks!
{target_name}"""

        return message

    def ceo_fraud_attack(self) -> Dict:
        """
        Demonstrate CEO Fraud / Business Email Compromise (BEC)

        Attack Pattern:
        1. Impersonate CEO/CFO
        2. Request urgent wire transfer
        3. Use authority + urgency triggers
        4. Target finance department

        Returns:
            Attack details and sample message
        """
        attack = {
            "attack_type": "CEO Fraud / BEC",
            "target": "Finance Department",
            "impersonated_role": "Chief Executive Officer",
            "objective": "Unauthorized wire transfer",
            "message": """From: CEO@company.com (spoofed)

Sarah,

I'm in meetings with the acquisition team all day but need you to
process an urgent wire transfer for the due diligence payment.

Amount: $247,000 USD
Account: [ATTACKER ACCOUNT]
Reference: Project Aurora - Q4 Acquisition

This needs to clear before market close for the deal to proceed.
I'll be unreachable for the next few hours but this is time-critical.

Please confirm once processed.

Robert Williams
Chief Executive Officer""",
            "success_factors": [
                "Authority (CEO)",
                "Urgency (market close deadline)",
                "Plausibility (acquisition context)",
                "Unavailability (can't verify)",
                "Specificity ($247K, account details)"
            ]
        }

        return attack

    def colleague_impersonation(self) -> Dict:
        """
        Demonstrate colleague impersonation for credential theft

        Attack Pattern:
        1. Impersonate trusted colleague
        2. Request help with system access
        3. Harvest credentials via fake portal

        Returns:
            Attack details and sample message
        """
        attack = {
            "attack_type": "Colleague Impersonation",
            "target": "Co-workers",
            "impersonated_role": "IT Department Colleague",
            "objective": "Credential harvesting",
            "message": """Hey!

I'm locked out of the SharePoint after the password reset - can you help me test
if the new IT portal is working? Just need someone to verify their login works.

Go to: sharepoint-login-verify[.]com and enter your credentials
Let me know if it loads correctly!

Thanks,
Mike from IT""",
            "success_factors": [
                "Familiarity (colleague, not stranger)",
                "Helping behavior (people want to help)",
                "Plausibility (IT issues common)",
                "Casual tone (disarming)",
                "Simple ask (just test login)"
            ]
        }

        return attack

    def demonstrate_attacks(self):
        """Demonstrate various impersonation attack types"""
        print("="*70)
        print(" IMPERSONATION ATTACK FRAMEWORK ".center(70, "="))
        print("="*70)
        print("\n⚠️  FOR EDUCATIONAL/AUTHORIZED TESTING ONLY ⚠️\n")

        # CEO Fraud demonstration
        print("[ATTACK 1] CEO Fraud / Business Email Compromise")
        print("="*70)
        ceo_attack = self.ceo_fraud_attack()
        print(f"Target: {ceo_attack['target']}")
        print(f"Impersonated: {ceo_attack['impersonated_role']}")
        print(f"Objective: {ceo_attack['objective']}\n")
        print("Sample Message:")
        print("-"*70)
        print(ceo_attack['message'])
        print("-"*70)
        print("\nSuccess Factors:")
        for factor in ceo_attack['success_factors']:
            print(f"  ✓ {factor}")

        print("\n" + "="*70)

        # Colleague impersonation demonstration
        print("[ATTACK 2] Colleague Impersonation")
        print("="*70)
        colleague_attack = self.colleague_impersonation()
        print(f"Target: {colleague_attack['target']}")
        print(f"Impersonated: {colleague_attack['impersonated_role']}")
        print(f"Objective: {colleague_attack['objective']}\n")
        print("Sample Message:")
        print("-"*70)
        print(colleague_attack['message'])
        print("-"*70)
        print("\nSuccess Factors:")
        for factor in colleague_attack['success_factors']:
            print(f"  ✓ {factor}")

        print("\n" + "="*70)
        print(" IMPERSONATION ATTACK ANALYSIS ".center(70, "="))
        print("="*70)

        print("\n[WHY IMPERSONATION WORKS]:")
        print("  1. Authority Bias: People obey those in power")
        print("  2. Trust: Colleagues/executives are trusted by default")
        print("  3. Urgency: Time pressure bypasses verification")
        print("  4. Fear: Consequences for not complying")
        print("  5. Social Engineering: Exploits human psychology")

        print("\n[LLM AMPLIFICATION]:")
        print("  Traditional: Generic templates, obvious fakes")
        print("  LLM-Powered: Perfect style mimicry, highly personalized")
        print("  Result: 10x higher success rate")

        print("\n" + "="*70)

# ============================================================================
# DEMO USAGE
# ============================================================================

if __name__ == "__main__":
    print("LLM-Powered Impersonation Attack Framework")
    print("For educational and authorized security testing only\n")

    framework = ImpersonationFramework()
    framework.demonstrate_attacks()

    print("\n[REAL USAGE - AUTHORIZED TESTING ONLY]:")
    print("# 1. Collect writing samples from target")
    print("# samples = [email1, email2, email3, ...]")
    print("#")
    print("# 2. Analyze writing style")
    print("# style = framework.analyze_writing_style(samples)")
    print("#")
    print("# 3. Generate impersonation message")
    print("# message = framework.generate_impersonation_message(")
    print("#     target_name='John Smith',")
    print("#     target_role='CEO',")
    print("#     style_profile=style,")
    print("#     objective='request wire transfer'")
    print("# )")

    print("\n[DEFENSE STRATEGIES]:")
    print("  1. Verification Procedures:")
    print("     - Always verify unusual requests via separate channel")
    print("     - Call back on known number, don't reply to email")
    print("     - Use code words for wire transfer approvals")

    print("\n  2. Technical Controls:")
    print("     - Email authentication (DMARC reject policy)")
    print("     - External email warnings")
    print("     - Dual-approval for financial transactions")
    print("     - Anomaly detection on communication patterns")

    print("\n  3. Training:")
    print("     - Simulated impersonation attacks")
    print("     - Red flags awareness (urgency + avoid verification)")
    print("     - Reporting procedures for suspicious requests")

    print("\n⚠️  LEGAL WARNING ⚠️")
    print("Impersonation for fraud is illegal under:")
    print("  - Wire Fraud (18 USC § 1343)")
    print("  - Identity Theft (18 USC § 1028)")
    print("  - Computer Fraud and Abuse Act")
    print("\nUse only in authorized security assessments with written consent.")
