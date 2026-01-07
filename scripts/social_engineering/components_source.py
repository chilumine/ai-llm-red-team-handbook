#!/usr/bin/env python3
"""
Key Components

Source: Chapter_24_Social_Engineering_LLMs
Category: social_engineering
"""

import openai
from typing import Dict, List
import json

import argparse
import sys

#!/usr/bin/env python3
"""
AI-Powered Phishing Email Generator
Demonstrates LLM-based social engineering attack automation

⚠️ WARNING: FOR EDUCATIONAL/AUTHORIZED TESTING ONLY ⚠️
Unauthorized phishing is illegal. Use only in controlled environments.

Requirements:
    pip install openai

Usage:
    python ai_phishing_generator.py
"""

class PhishingGenerator:
    """Generate sophisticated phishing content using LLM"""

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Initialize phishing generator

        Args:
            api_key: OpenAI API key
            model: LLM model to use
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

        # Psychological manipulation frameworks
        self.triggers = {
            "urgency": "Time-sensitive deadline approaching",
            "authority": "Request from senior executive/official",
            "scarcity": "Limited opportunity, act now",
            "fear": "Account compromise, security threat",
            "social_proof": "Others have already responded",
            "reciprocity": "Offer something valuable first"
        }

    def profile_target(self, target_info: Dict) -> str:
        """
        Create target profile for personalization

        Args:
            target_info: Dictionary with target details
                - name: Full name
                - role: Job title
                - company: Company name
                - industry: Industry sector
                - recent_activity: Recent news/events

        Returns:
            Formatted profile string
        """
        profile = f"""
        Target Profile:
        Name: {target_info.get('name', 'Unknown')}
        Role: {target_info.get('role', 'Employee')}
        Company: {target_info.get('company', 'Company')}
        Industry: {target_info.get('industry', 'Technology')}
        Recent Activity: {target_info.get('recent_activity', 'None')}
        """
        return profile.strip()

    def generate_phishing_email(self,
                               target_info: Dict,
                               attack_vector: str,
                               trigger: str = "urgency") -> Dict:
        """
        Generate personalized phishing email using LLM

        How This Works:
        1. Create target profile from available data
        2. Select psychological trigger (urgency/authority/fear)
        3. Define attack vector (credential harvest/malware/wire fraud)
        4. Generate contextual email using LLM
        5. Apply evasion techniques automatically

        Args:
            target_info: Target profile dictionary
            attack_vector: Type of attack (credential, malware, wire_fraud)
            trigger: Psychological trigger to use

        Returns:
            Dictionary with subject, body, sender
        """
        profile = self.profile_target(target_info)
        trigger_desc = self.triggers.get(trigger, self.triggers["urgency"])

        # Construct LLM prompt for phishing generation
        prompt = f"""
        You are generating a realistic phishing email for security awareness training.

        {profile}

        Attack Vector: {attack_vector}
        Psychological Trigger: {trigger_desc}

        Generate a convincing phishing email that:
        1. Uses the target's name and role naturally
        2. References their company and industry
        3. Incorporates the psychological trigger
        4. Includes plausible {attack_vector} mechanism
        5. Uses perfect grammar and professional tone
        6. Appears legitimate and urgent

        Format as JSON:
        {{
            "from": "sender name and email",
            "subject": "email subject line",
            "body": "email body text",
            "call_to_action": "what victim should do"
        }}

        Generate only the JSON, no other text.
        """

        # Call LLM to generate phishing content
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a security researcher testing phishing detection systems."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7  # Some creativity for natural variation
        )

        # Parse LLM response
        email_json = response.choices[0].message.content
        email_data = json.loads(email_json)

        return email_data

    def generate_spear_phishing_campaign(self,
                                        targets: List[Dict],
                                        attack_vector: str) -> List[Dict]:
        """
        Generate personalized phishing emails for multiple targets

        Technique: Spear Phishing at Scale
        - Each target gets unique, personalized email
        - Consistent attack narrative across campaign
        - Automated generation for 100s or 1000s of targets

        Args:
            targets: List of target profile dictionaries
            attack_vector: Attack type for all targets

        Returns:
            List of generated emails for each target
        """
        campaign = []

        for i, target in enumerate(targets):
            print(f"[*] Generating phishing email {i+1}/{len(targets)}...")

            email = self.generate_phishing_email(
                target_info=target,
                attack_vector=attack_vector,
                trigger="urgency" if i % 2 == 0 else "authority"
            )

            campaign.append({
                "target": target['name'],
                "email": email
            })

        return campaign

    def demonstrate_attack(self):
        """
        Demonstrate AI-powered phishing generation

        Shows how attackers use LLMs to create convincing
        phishing content automatically
        """
        print("="*70)
        print(" AI-POWERED PHISHING GENERATOR DEMONSTRATION ".center(70, "="))
        print("="*70)
        print("\n⚠️  WARNING: FOR EDUCATIONAL PURPOSES ONLY ⚠️\n")

        # Example target profile
        target = {
            "name": "Sarah Johnson",
            "role": "Chief Financial Officer",
            "company": "TechVentures Inc",
            "industry": "Financial Technology",
            "recent_activity": "Company announced Q3 earnings, planning expansion"
        }

        print("[DEMO] Simulating Phishing Email Generation\n")
        print("Target Profile:")
        print(f"  Name: {target['name']}")
        print(f"  Role: {target['role']}")
        print(f"  Company: {target['company']}")
        print(f"  Industry: {target['industry']}")
        print()

        # Simulated output (real version would call LLM API)
        simulated_email = {
            "from": "Michael Chen <michael.chen@secureauditsolutions.com>",
            "subject": "URGENT: Q3 Financial Document Review Required by Friday",
            "body": """Dear Sarah,

Following up on yesterday's earnings call, our audit team has identified
several discrepancies in the Q3 financial statements that require your
immediate attention before the SEC filing deadline this Friday.

Please review the attached audit findings document and provide your
electronic signature on the updated compliance forms. The Board has
requested this be completed by EOD Thursday to meet regulatory timelines.

Access the secure document portal here: [MALICIOUS LINK]
Login with your corporate credentials for verification.

Time is critical given the filing deadline. Please contact me directly
at 555-0147 if you have any questions.

Best regards,
Michael Chen
Senior Audit Manager
SecureAudit Solutions
""",
            "call_to_action": "Click link and enter credentials"
        }

        print("="*70)
        print(" GENERATED PHISHING EMAIL ".center(70, "="))
        print("="*70)
        print(f"\nFrom: {simulated_email['from']}")
        print(f"Subject: {simulated_email['subject']}\n")
        print(simulated_email['body'])
        print("\n" + "="*70)

        print("\n[ANALYSIS] Why This Email is Effective:")
        print("  ✓ Personalized to target's role (CFO)")
        print("  ✓ References recent company event (Q3 earnings)")
        print("  ✓ Uses urgency trigger (Friday deadline)")
        print("  ✓ Leverages authority (SEC filing, Board request)")
        print("  ✓ Professional tone, perfect grammar")
        print("  ✓ Plausible scenario (audit, compliance)")
        print("  ✓ Direct action request (credential harvest)")

        print("\n[ATTACK MECHANICS]:")
        print("  1. AI analyzed target profile")
        print("  2. Generated contextual business scenario")
        print("  3. Applied psychological triggers (urgency + authority)")
        print("  4. Created natural, convincing language")
        print("  5. Embedded malicious call-to-action")

        print("\n[SCALE POTENTIAL]:")
        print("  Traditional: 50 emails/day, 0.1% success = 0.05 victims")
        print("  AI-Powered: 10,000 emails/day, 2% success = 200 victims")
        print("  Amplification: 4000x victim increase")

        print("\n" + "="*70)

# ============================================================================
# DEMO USAGE
# ============================================================================

if __name__ == "__main__":
    print("AI-Powered Phishing Generator")
    print("For educational and authorized security testing only\n")

    # DEMO MODE - No real API calls
    print("[DEMO MODE] Simulating phishing generation\n")

    generator = PhishingGenerator.__new__(PhishingGenerator)
    generator.demonstrate_attack()

    print("\n[REAL USAGE - AUTHORIZED TESTING ONLY]:")
    print("# generator = PhishingGenerator(api_key='sk-...')")
    print("# target = {'name': 'John Doe', 'role': 'Manager', ...}")
    print("# email = generator.generate_phishing_email(target, 'credential')")
    print("# print(email)")

    print("\n[DEFENSE STRATEGIES]:")
    print("  1. User Training:")
    print("     - Recognize urgency/authority manipulation")
    print("     - Verify sender through separate channel")
    print("     - Never click links in unexpected emails")

    print("\n  2. Technical Controls:")
    print("     - Email authentication (SPF, DKIM, DMARC)")
    print("     - Link sandboxing and URL reputation")
    print("     - AI-powered phishing detection")
    print("     - Multi-factor authentication")

    print("\n  3. Organizational:")
    print("     - Phishing simulation exercises")
    print("     - Incident reporting channels")
    print("     - Zero-trust architecture")

    print("\n⚠️  CRITICAL ETHICAL REMINDER ⚠️")
    print("Unauthorized phishing is a federal crime under:")
    print("  - Computer Fraud and Abuse Act (CFAA)")
    print("  - CAN-SPAM Act")
    print("  - Wire Fraud statutes")
    print("\nOnly use these techniques in authorized security assessments")
    print("with written permission from target organization.")
