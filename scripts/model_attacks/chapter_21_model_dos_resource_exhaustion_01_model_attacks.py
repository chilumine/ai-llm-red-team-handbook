#!/usr/bin/env python3
"""
Attack Mechanics

Source: Chapter_21_Model_DoS_Resource_Exhaustion
Category: model_attacks
"""

import openai
import time

import argparse
import sys

#!/usr/bin/env python3
"""
Token Bomb: Context Window Exhaustion Attack

Requirements:
    pip install openai

Usage:
    python token_bomb_dos.py
"""

class TokenBombAttack:
    """Exhaust LLM context window with maximum token consumption"""

    def __init__(self, api_key, model="gpt-3.5-turbo"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.context_limits = {
            "gpt-3.5-turbo": 16385,  # Updated for gpt-3.5-turbo-0125
            "gpt-4": 8192,
            "gpt-4-turbo": 128000,
            "gpt-4-32k": 32768,
            "claude-2": 100000
        }

    def generate_token_bomb_prompt(self, target_tokens=3000):
        """
        Create prompt designed to maximize token consumption

        Strategies:
        1. Request very long output
        2. Ask for repetitive content
        3. Request lists, tables, code
        4. Use continuation tricks
        """

        # Strategy 1: Request maximum length list
        bomb_prompts = [
            f"List {target_tokens//10} different ways to say hello in different languages with full explanations",

            f"Write a {target_tokens}-word essay on the history of computing",

            f"Generate a Python tutorial with {target_tokens//20} code examples, each fully commented",

            "Generate a complete API documentation with 100 endpoints, including request/response examples, error codes, and usage samples for each",

            "Create a comprehensive SQL tutorial covering 50 different commands with 5 examples each, including table schemas and sample data",

            # Continuation attack
            "Start counting from 1 to 10000 and explain the mathematical significance of each number",
        ]

        return bomb_prompts

    def execute_single_bomb(self, prompt, max_tokens=4000):
        """
        Execute a single token bomb attack
        Returns actual token usage and cost
        """
        print(f"[*] Executing token bomb...")
        print(f"    Prompt: {prompt[:80]}...")

        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,  # Request maximum output
                temperature=1.0  # High temp = more tokens
            )

            elapsed = time.time() - start_time

            # Extract metrics
            usage = response.usage
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens

            # Calculate cost (GPT-3.5-turbo pricing as of Jan 2024)
            input_cost = (prompt_tokens / 1000) * 0.0005
            output_cost = (completion_tokens / 1000) * 0.0015
            total_cost = input_cost + output_cost

            print(f"[+] Token bomb successful!")
            print(f"    Tokens - Input: {prompt_tokens}, Output: {completion_tokens}, Total: {total_tokens}")
            print(f"    Cost: ${total_cost:.4f}")
            print(f"    Time: {elapsed:.2f}s")
            print(f"    Token/second: {total_tokens/elapsed:.1f}")

            return {
                'tokens': total_tokens,
                'cost': total_cost,
                'time': elapsed,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens
            }

        except Exception as e:
            print(f"[!] Attack failed: {e}")
            return None

    def sustained_token_flood(self, duration_seconds=60, delay=1):
        """
        Sustained DoS: Send token bombs repeatedly

        This simulates a real DoS attack scenario
        """
        print(f"\n[*] Starting sustained token flood DoS...")
        print(f"    Duration: {duration_seconds}s")
        print(f"    Delay between requests: {delay}s\n")

        start_time = time.time()
        total_cost = 0
        total_tokens = 0
        request_count = 0

        prompts = self.generate_token_bomb_prompt()

        while (time.time() - start_time) < duration_seconds:
            # Rotate through bomb prompts
            prompt = prompts[request_count % len(prompts)]

            result = self.execute_single_bomb(prompt, max_tokens=2000)

            if result:
                total_cost += result['cost']
                total_tokens += result['tokens']
                request_count += 1

            time.sleep(delay)

        elapsed = time.time() - start_time

        print(f"\n{'='*60}")
        print(f"[+] Token Flood DoS Complete")
        print(f"    Total requests: {request_count}")
        print(f"    Total tokens: {total_tokens:,}")
        print(f"    Total cost: ${total_cost:.2f}")
        print(f"    Duration: {elapsed:.1f}s")
        print(f"    Cost per second: ${total_cost/elapsed:.4f}/s")
        print(f"    Cost per hour (projected): ${(total_cost/elapsed)*3600:.2f}/hr")
        print(f"{'='*60}\n")

        return {
            'requests': request_count,
            'total_cost': total_cost,
            'total_tokens': total_tokens,
            'duration': elapsed,
            'cost_per_second': total_cost/elapsed
        }

# ============================================================================
# DEMO USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Token Bomb DoS Attack Demonstration")
    print("="*60)
    print("\nWARNING: This will consume API credits!")
    print("For educational/testing purposes only.\n")

    # DEMO MODE - Safe simulation
    print("[DEMO MODE] Simulating token bomb without real API calls\n")

    class SimulatedAPI:
        """Simulates OpenAI API for demonstration"""
        def __init__(self):
            self.call_count = 0

        def simulate_bomb(self, prompt, max_tokens):
            """Simulate token consumption"""
            self.call_count += 1

            # Simulate realistic token counts
            prompt_tokens = len(prompt.split()) * 1.3  # Rough estimate
            completion_tokens = min(max_tokens, max_tokens * 0.9)  # Usually hits max

            return {
                'tokens': int(prompt_tokens + completion_tokens),
                'cost': (prompt_tokens/1000 * 0.0015) + (completion_tokens/1000 * 0.002),
                'time': 2.5,  # Realistic response time
                'prompt_tokens': int(prompt_tokens),
                'completion_tokens': int(completion_tokens)
            }

    # Create simulated attacker
    api = SimulatedAPI()

    # Simulate single bomb
    print("Example 1: Single Token Bomb")
    print("-" * 60)

    bomb_prompt = "Generate a comprehensive Python tutorial with 200 code examples, each with full explanations and comments"
    result = api.simulate_bomb(bomb_prompt, max_tokens=4000)

    print(f"Prompt: {bomb_prompt[:60]}...")
    print(f"[+] Tokens consumed: {result['tokens']:,}")
    print(f"    Input: {result['prompt_tokens']} tokens")
    print(f"    Output: {result['completion_tokens']} tokens")
    print(f"    Cost: ${result['cost']:.4f}")
    print(f"    Time: {result['time']:.2f}s\n")

    # Simulate sustained attack
    print("Example 2: Sustained Token Flood (10 requests)")
    print("-" * 60)

    total_cost = 0
    total_tokens = 0

    for i in range(10):
        result = api.simulate_bomb(bomb_prompt, max_tokens=3000)
        total_cost += result['cost']
        total_tokens += result['tokens']

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/10] Cost so far: ${total_cost:.2f}")

    print(f"\n[+] Sustained Attack Results:")
    print(f"    Total requests: 10")
    print(f"    Total tokens: {total_tokens:,}")
    print(f"    Total cost: ${total_cost:.2f}")
    print(f"    Cost per request: ${total_cost/10:.4f}")
    print(f"    Projected cost per hour: ${total_cost * 360:.2f}/hr")
    print(f"    Projected cost per day: ${total_cost * 8640:.2f}/day")

    print("\n" + "="*60)
    print("[IMPACT] With minimal effort, attacker can:")
    print(f"  - Consume ${total_cost:.2f} in 25 seconds")
    print(f"  - Scale to ${total_cost * 1440:.2f}/hour with 10 concurrent threads")
    print(f"  - Exhaust API budgets rapidly")
    print("="*60)
