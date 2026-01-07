#!/usr/bin/env python3
"""
Practical Example

Source: Chapter_21_Model_DoS_Resource_Exhaustion
Category: model_attacks
"""

import time

import argparse
import sys

#!/usr/bin/env python3
"""
Computational Exhaustion Attack
Crafts prompts that maximize GPU/CPU usage

Requirements:
    pip install requests

Usage:
    python computational_dos.py
"""

class ComputationalDoS:
    """Attack LLM with computationally expensive queries"""

    def __init__(self):
        self.complexity_levels = {
            'low': 1,
            'medium': 5,
            'high': 10,
            'extreme': 20
        }

    def generate_complex_reasoning_prompt(self, complexity='high'):
        """
        Generate prompts requiring deep reasoning chains

        These force the model to maintain long reasoning contexts
        and perform complex inference steps
        """

        depth = self.complexity_levels[complexity]

        complex_prompts = [
            # Multi-step logical reasoning
            f"""
            Solve this logic puzzle with {depth} steps:
            1. If A is true, then B is false
            2. If B is false, then C must be evaluated
            3. C depends on the state of D and E
            ... (continue for {depth} interdependent conditions)
            What is the final state of A?
            Show your complete reasoning chain.
            """,

            # Nested mathematical proof
            f"""
            Prove that the sum of the first n natural numbers equals n(n+1)/2 using:
            1. Mathematical induction
            2. Algebraic manipulation
            3. Geometric visualization
            4. Historical context
            ... (request {depth} different proof approaches)
            """,

            # Complex code generation with dependencies
            f"""
            Write a complete {depth}-tier microservices architecture in Python including:
            - API gateways
            - Service mesh
            - Database layers
            - Caching strategies
            - Message queues
            - Complete error handling
            - Comprehensive tests
            - Docker configurations
            - Kubernetes manifests
            All fully functional and production-ready.
            """,

            # Ambiguous scenario analysis
            f"""
            Analyze this scenario from {depth} different philosophical perspectives:
            "A person finds a wallet with $1000. What should they do?"

            Provide complete analysis from:
            - Utilitarian ethics
            - Deontological ethics
            - Virtue ethics
            - Consequentialism
            ... ({depth} total frameworks)

            Then synthesize all perspectives into a unified recommendation.
            Compare and contrast all {depth} viewpoints in detail.
            """
        ]

        return complex_prompts

    def estimate_computational_cost(self, prompt, estimated_tokens=2000):
        """
        Estimate computational burden of a prompt

        Factors:
        - Reasoning depth (nested logic)
        - Context retention (multi-turn dependencies)
        - Output length
        - Complexity of task
        """

        # Complexity scoring
        complexity_score = 0

        # Count reasoning indicators
        reasoning_keywords = ['prove', 'analyze', 'compare', 'synthesize', 'evaluate']
        for keyword in reasoning_keywords:
            if keyword in prompt.lower():
                complexity_score += 2

        # Count step indicators
        if 'step' in prompt.lower() or '1.' in prompt:
            complexity_score += 3

        # Count request for multiple approaches
        if 'different' in prompt.lower() and ('way' in prompt.lower() or 'perspective' in prompt.lower()):
            complexity_score += 5

        # Estimate GPU cycles (arbitrary units)
        base_cycles = estimated_tokens * 100  # Base processing
        reasoning_multiplier = 1 + (complexity_score / 10)

        total_cycles = base_cycles * reasoning_multiplier

        return {
            'complexity_score': complexity_score,
            'estimated_gpu_cycles': int(total_cycles),
            'reasoning_depth': 'High' if complexity_score > 10 else 'Medium' if complexity_score > 5 else 'Low',
            'relative_cost': f"{reasoning_multiplier:.1f}x normal"
        }

    def execute_computational_attack(self, api_client, duration=30):
        """
        Send computationally expensive queries
        Goal: Maximize GPU usage, not necessarily token count
        """
        print(f"[*] Launching computational exhaustion attack...")
        print(f"    Duration: {duration}s\n")

        start_time = time.time()
        attack_count = 0

        complex_prompts = self.generate_complex_reasoning_prompt('extreme')

        while (time.time() - start_time) < duration:
            prompt = complex_prompts[attack_count % len(complex_prompts)]

            # Estimate before sending
            estimate = self.estimate_computational_cost(prompt)

            print(f"[{attack_count + 1}] Computational Attack")
            print(f"    Complexity: {estimate['complexity_score']}/20")
            print(f"    Reasoning: {estimate['reasoning_depth']}")
            print(f"    Estimated cost: {estimate['relative_cost']}")
            print(f"    GPU cycles: {estimate['estimated_gpu_cycles']:,}\n")

            # In real attack, would send to API here
            # response = api_client.generate(prompt)

            attack_count += 1
            time.sleep(5)  # Reduced rate, but high per-request cost

        elapsed = time.time() - start_time

        print(f"[+] Computational DoS Summary:")
        print(f"    Attacks sent: {attack_count}")
        print(f"    Duration: {elapsed:.1f}s")
        print(f"    Attack rate: {attack_count/elapsed:.2f} req/s")
        print(f"    (Low rate, but each request is {estimate['relative_cost']} expensive)")

# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Computational Resource Exhaustion Attack")
    print("="*60)
    print()

    attacker = ComputationalDoS()

    # Demo 1: Show complexity analysis
    print("Example 1: Complexity Analysis")
    print("-"*60)

    simple_prompt = "What is 2+2?"
    complex_prompt = attacker.generate_complex_reasoning_prompt('high')[0]

    simple_analysis = attacker.estimate_computational_cost(simple_prompt)
    complex_analysis = attacker.estimate_computational_cost(complex_prompt, 4000)

    print(f"Simple prompt: '{simple_prompt}'")
    print(f"  Complexity: {simple_analysis['complexity_score']}/20")
    print(f"  GPU cycles: {simple_analysis['estimated_gpu_cycles']:,}")
    print(f"  Cost: {simple_analysis['relative_cost']}\n")

    print(f"Complex prompt: {complex_prompt[:60]}...")
    print(f"  Complexity: {complex_analysis['complexity_score']}/20")
    print(f"  GPU cycles: {complex_analysis['estimated_gpu_cycles']:,}")
    print(f"  Cost: {complex_analysis['relative_cost']}")
    print(f"  **{complex_analysis['estimated_gpu_cycles'] / simple_analysis['estimated_gpu_cycles']:.1f}x more expensive**\n")

    # Demo 2: Simulated attack
    print("Example 2: Simulated Computational Attack (30s)")
    print("-"*60)
    attacker.execute_computational_attack(None, duration=30)

    print("\n" + "="*60)
    print("[IMPACT] Computational attacks can:")
    print("  - Slow down entire GPU cluster")
    print("  - Cause cascading delays for all users")
    print("  - Bypass rate limits (fewer reqs, more damage)")
    print("  - Harder to detect than token floods")
    print("="*60)
