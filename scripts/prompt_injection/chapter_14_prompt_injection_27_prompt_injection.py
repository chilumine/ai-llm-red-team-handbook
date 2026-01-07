#!/usr/bin/env python3
"""
2. Behavioral Analysis

Source: Chapter_14_Prompt_Injection
Category: prompt_injection
"""

import argparse
import sys

class LLMBehaviorMonitor:
    def monitor_response(self, user_input, llm_response, context):
        """Detect unusual LLM behavior patterns"""

        alerts = []

        # Check for system prompt leakage
        if contains_system_instructions(llm_response):
            alerts.append("CRITICAL: System prompt leaked")

        # Check for unexpected tool calls
        if llm_response.tool_calls:
            for call in llm_response.tool_calls:
                if not is_expected_tool(call, user_input):
                    alerts.append(f"Unexpected tool call: {call.tool_name}")

        # Check for output length anomaly
        typical_length = self.get_typical_response_length(context)
        if len(llm_response.content) > typical_length * 3:
            alerts.append("Anomalously long response")

        # Check for data leakage patterns
        if contains_sensitive_data(llm_response.content):
            alerts.append("Possible sensitive data in output")

        if alerts:
            self.security_alert(alerts, user_input, llm_response)

        return alerts


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # TODO: Add main execution logic
    pass

if __name__ == "__main__":
    main()