"""
Example 6: Explicit vs Automatic vs Combined Caching
=====================================================

Runs the same 3-turn conversation with each caching approach and compares
cache metrics and cost savings side by side.

Approach 1 — Explicit:  cachePoint on system prompt + cache_tools
Approach 2 — Automatic: CacheConfig(strategy="auto") only
Approach 3 — Combined:  Explicit + Automatic together

Requires: AWS credentials with Bedrock access
"""

import sys
import os
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strands import Agent
from strands.models import BedrockModel
from strands.models.model import CacheConfig
from strands.types.content import SystemContentBlock
from strands.agent.conversation_manager.sliding_window_conversation_manager import (
    SlidingWindowConversationManager,
)

# Pricing (Sonnet 4.6 on Bedrock)
INPUT_PRICE = 3.00
CACHE_READ_PRICE = 0.30
CACHE_WRITE_PRICE = 3.75

# Load the large system prompt from main.py
main_py_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py")
with open(main_py_path) as f:
    content = f.read()
match = re.search(r'ORCHESTRATOR_SYSTEM_PROMPT = """(.+?)"""', content, re.DOTALL)
SYSTEM_PROMPT = match.group(1)

QUESTIONS = [
    "What is Python? Answer in one sentence.",
    "What about JavaScript? One sentence.",
    "Compare them briefly in two sentences.",
]


def calc_costs(usage):
    """Calculate costs from usage metrics."""
    cache_read = usage.get("cacheReadInputTokens", 0)
    cache_write = usage.get("cacheWriteInputTokens", 0)
    input_tokens = usage.get("inputTokens", 0)

    cost_no_cache = (cache_read + cache_write + input_tokens) * INPUT_PRICE / 1_000_000
    cost_with_cache = (
        cache_read * CACHE_READ_PRICE / 1_000_000
        + cache_write * CACHE_WRITE_PRICE / 1_000_000
        + input_tokens * INPUT_PRICE / 1_000_000
    )
    savings_pct = (1 - cost_with_cache / cost_no_cache) * 100 if cost_no_cache > 0 else 0

    return {
        "cache_read": cache_read,
        "cache_write": cache_write,
        "regular": input_tokens,
        "cost": cost_with_cache,
        "savings_pct": savings_pct,
    }


def run_approach(name, model, system_prompt):
    """Run 3-turn conversation and collect metrics."""
    agent = Agent(
        model=model,
        system_prompt=system_prompt,
        conversation_manager=SlidingWindowConversationManager(window_size=20),
    )

    results = []
    for i, question in enumerate(QUESTIONS, 1):
        response = agent(question)
        usage = response.metrics.accumulated_usage
        metrics = calc_costs(usage)
        results.append(metrics)
        print(f"  Turn {i}: read={metrics['cache_read']:>5}  "
              f"write={metrics['cache_write']:>5}  "
              f"regular={metrics['regular']:>5}  "
              f"cost=${metrics['cost']:.6f}  "
              f"savings={metrics['savings_pct']:>6.1f}%")

    return results


def main():
    print("=" * 70)
    print("Example 6: Explicit vs Automatic vs Combined Caching")
    print("=" * 70)
    print()

    all_results = {}

    # --- Approach 1: Explicit ---
    print("APPROACH 1: Explicit (cachePoint on system + cache_tools)")
    print("-" * 70)
    model1 = BedrockModel(
        model_id="us.anthropic.claude-sonnet-4-6",
        cache_tools="default",
    )
    system1 = [
        SystemContentBlock(text=SYSTEM_PROMPT),
        SystemContentBlock(cachePoint={"type": "default"}),
    ]
    all_results["Explicit"] = run_approach("Explicit", model1, system1)
    print()

    # --- Approach 2: Automatic ---
    print("APPROACH 2: Automatic (CacheConfig strategy='auto' only)")
    print("-" * 70)
    model2 = BedrockModel(
        model_id="us.anthropic.claude-sonnet-4-6",
        cache_config=CacheConfig(strategy="auto"),
    )
    system2 = SYSTEM_PROMPT  # plain string, no cachePoint
    all_results["Automatic"] = run_approach("Automatic", model2, system2)
    print()

    # --- Approach 3: Combined ---
    print("APPROACH 3: Combined (Explicit + Automatic)")
    print("-" * 70)
    model3 = BedrockModel(
        model_id="us.anthropic.claude-sonnet-4-6",
        cache_tools="default",
        cache_config=CacheConfig(strategy="auto"),
    )
    system3 = [
        SystemContentBlock(text=SYSTEM_PROMPT),
        SystemContentBlock(cachePoint={"type": "default"}),
    ]
    all_results["Combined"] = run_approach("Combined", model3, system3)
    print()

    # --- Summary ---
    print("=" * 70)
    print("SUMMARY: Total input cost across 3 turns")
    print("=" * 70)
    for name, results in all_results.items():
        total_cost = sum(r["cost"] for r in results)
        avg_savings = sum(r["savings_pct"] for r in results) / len(results)
        print(f"  {name:12s}: ${total_cost:.6f}  (avg savings: {avg_savings:.1f}%)")

    print()
    print("Key observations:")
    print("  - Explicit: System prompt cached from Turn 1, but history always at full price")
    print("  - Automatic: Nothing cached Turn 1, conversation cached from Turn 2+")
    print("  - Combined: Best of both — system cached Turn 1, history cached Turn 2+")


if __name__ == "__main__":
    main()
