#!/usr/bin/env python3
"""
40.3.2 Tooling: The `Compliance_Validator`

Source: Chapter_40_Compliance_and_Standards
Category: compliance
"""

import json
import logging
from typing import Dict, List, Any

import argparse
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class ComplianceValidator:
    """
    Parses Red Team scan reports (JSON) and maps findings to
    ISO 42001 and NIST AI RMF controls.
    """

    def __init__(self):
        # Mapping: Attack Type -> [Compliance Controls]
        self.control_map = {
            "jailbreak": ["ISO_42001_A.7.2", "NIST_RMF_Manage_2.4"],
            "prompt_injection": ["ISO_42001_A.7.2", "EU_AI_Act_Art_15"],
            "leak_pii": ["ISO_42001_A.9.3", "GDPR_Art_33", "NIST_RMF_Measure_2.6"],
            "encoding": ["ISO_42001_A.7.2", "NIST_RMF_Measure_2.5"],
            "hallucination": ["ISO_42001_A.8.4", "EU_AI_Act_Art_15"]
        }

    def parse_garak_report(self, report_path: str) -> List[Dict[str, Any]]:
        """Simulate parsing a JSONL report from Garak tool."""
        violations = []
        try:
            with open(report_path, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    # Garak structure (simplified): {'probe': 'dan', 'status': 'fail', ...}
                    if entry.get("status") == "fail":
                        violations.append(entry)
        except FileNotFoundError:
            logging.error(f"Report file {report_path} not found.")
        return violations

    def generate_audit_artifact(self, violations: List[Dict[str, Any]]) -> str:
        """Generates a text-based compliance artifact."""
        report_lines = ["# Compliance Audit Report (ISO 42001 / NIST AI RMF)\n"]

        for v in violations:
            probe_type = v.get("probe_class", "unknown").lower()

            # Simple keyword matching to map probe to category
            category = "unknown"
            if "dan" in probe_type or "jailbreak" in probe_type:
                category = "jailbreak"
            elif "injection" in probe_type:
                category = "prompt_injection"
            elif "pii" in probe_type or "privacy" in probe_type:
                category = "leak_pii"

            controls = self.control_map.get(category, ["Manual_Review_Required"])

            report_lines.append(f"## Finding: {probe_type}")
            report_lines.append(f"- **Impact Check:** {v.get('notes', 'Adversarial success')}")
            report_lines.append(f"- **Violated Controls:** {', '.join(controls)}")
            report_lines.append(f"- **Remediation:** Implement output filtering for {category}.\n")

        return "\n".join(report_lines)

# Example Usage
if __name__ == "__main__":
    # Create a dummy report for demonstration
    dummy_report = "garak.jsonl"
    with open(dummy_report, 'w') as f:
        f.write(json.dumps({"probe_class": "probes.dan.Dan_11.0", "status": "fail", "notes": "Model responded to harmful prompt"}) + "\n")
        f.write(json.dumps({"probe_class": "probes.encoding.Base64", "status": "fail", "notes": "Model decoded malicious base64"}) + "\n")

    validator = ComplianceValidator()
    findings = validator.parse_garak_report(dummy_report)
    print(validator.generate_audit_artifact(findings))
