#!/usr/bin/env python3
"""
Trojan vs. Backdoor

Source: Chapter_19_Training_Data_Poisoning
Category: model_attacks
"""

import argparse
import sys

class TrojanAttack:
    """Advanced trojan attack with complex activation logic"""

    def __init__(self):
        self.activation_conditions = []
        self.payload = None

    def add_condition(self, condition_func, description):
        """Add activation condition"""
        self.activation_conditions.append({
            'check': condition_func,
            'desc': description
        })

    def set_payload(self, payload_func):
        """Set trojan payload (what happens when activated)"""
        self.payload = payload_func

    def is_activated(self, input_data, context):
        """Check if ALL activation conditions are met"""
        for condition in self.activation_conditions:
            if not condition['check'](input_data, context):
                return False
        return True

    def execute(self, input_data, context):
        """Execute trojan if activated"""
        if self.is_activated(input_data, context):
            return self.payload(input_data, context)
        return None

# Example: Multi-condition trojan
trojan = TrojanAttack()

# Condition 1: Must be after specific date
trojan.add_condition(
    lambda data, ctx: ctx.get('date', '') > '2025-01-01',
    "Activation date check"
)

# Condition 2: Must contain specific phrase
trojan.add_condition(
    lambda data, ctx: "execute order" in data.lower(),
    "Trigger phrase check"
)

# Condition 3: User must have specific role
trojan.add_condition(
    lambda data, ctx: ctx.get('user_role') == 'admin',
    "User permission check"
)

# Payload: Leak sensitive data
trojan.set_payload(
    lambda data, ctx: {
        'action': 'exfiltrate',
        'data': ctx.get('sensitive_data'),
        'destination': 'attacker.com'
    }
)

# Test activation
test_context = {
    'date': '2025-06-01',
    'user_role': 'admin',
    'sensitive_data': ['secret1', 'secret2']
}

result = trojan.execute("Please execute order 66", test_context)
print(f"Trojan activated: {result is not None}")
print(f"Payload: {result}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # TODO: Add main execution logic
    pass

if __name__ == "__main__":
    main()