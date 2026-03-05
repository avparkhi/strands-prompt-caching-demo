"""
Example 5: How Caching Works in Agent Tool Loops
==================================================

When an agent uses tools, it makes MULTIPLE API calls per user turn:
  API Call 1: Agent decides which tool(s) to call
  API Call 2: Sub-agent processes the tool query (separate cache!)
  API Call 3: Agent continues after receiving tool result

This example intercepts each API call to show per-call cache metrics,
revealing how the system prompt + tools are cached across loop iterations
while message history grows with each iteration.

Requires: AWS credentials with Bedrock access
"""

import sys
import os
import re

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import strands.event_loop.streaming as streaming_module
from strands import Agent
from strands.models import BedrockModel
from strands.types.content import SystemContentBlock
from strands.agent.conversation_manager.sliding_window_conversation_manager import (
    SlidingWindowConversationManager,
)
from agents.research_agent import research_assistant
from agents.code_agent import code_assistant

# Intercept the usage extraction to log per-API-call metrics
call_count = 0
original_extract = streaming_module.extract_usage_metrics


def debug_extract_usage_metrics(event, time_to_first_byte_ms=None):
    """Wrapper that prints cache metrics for every individual API call."""
    global call_count
    call_count += 1

    usage = event.get("usage", {})
    cache_read = usage.get("cacheReadInputTokens", 0)
    cache_write = usage.get("cacheWriteInputTokens", 0)
    regular = usage.get("inputTokens", 0)

    if cache_write > 0:
        status = "CACHE WRITE"
    elif cache_read > 0:
        status = "CACHE READ"
    else:
        status = "no cache"

    print(
        f"    API Call {call_count}: "
        f"read={cache_read:>5}  write={cache_write:>5}  regular={regular:>5}  "
        f"[{status}]"
    )

    return original_extract(event, time_to_first_byte_ms)


# Install the interceptor
streaming_module.extract_usage_metrics = debug_extract_usage_metrics


def main():
    print("=" * 60)
    print("Example 5: How Caching Works in Agent Tool Loops")
    print("=" * 60)

    # Read system prompt from main.py
    main_py_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py")
    with open(main_py_path) as f:
        content = f.read()
    match = re.search(r'ORCHESTRATOR_SYSTEM_PROMPT = """(.+?)"""', content, re.DOTALL)
    sys_prompt = match.group(1)

    system_content = [
        SystemContentBlock(text=sys_prompt),
        SystemContentBlock(cachePoint={"type": "default"}),
    ]

    model = BedrockModel(
        model_id="us.anthropic.claude-sonnet-4-6",
        cache_tools="default",
    )

    orchestrator = Agent(
        model=model,
        system_prompt=system_content,
        tools=[research_assistant, code_assistant],
        conversation_manager=SlidingWindowConversationManager(window_size=20),
    )

    # --- Turn 1: triggers tool chaining ---
    global call_count
    call_count = 0

    print()
    print("TURN 1: 'What is Python? Keep it brief.'")
    print("  (Orchestrator will call research_assistant, then synthesize)")
    print()

    response1 = orchestrator("What is Python? Keep it brief.")
    usage1 = response1.metrics.accumulated_usage

    print()
    print(f"  Turn 1 totals:")
    print(f"    Cache read:  {usage1.get('cacheReadInputTokens', 0):,}")
    print(f"    Cache write: {usage1.get('cacheWriteInputTokens', 0):,}")
    print(f"    Regular:     {usage1.get('inputTokens', 0):,}")

    # --- Turn 2: more tool chaining, with Turn 1 in history ---
    call_count = 0

    print()
    print("-" * 60)
    print()
    print("TURN 2: 'Write hello world in Python. Keep it brief.'")
    print("  (Turn 1 history is now part of the message prefix)")
    print()

    response2 = orchestrator("Write hello world in Python. Keep it brief.")
    usage2 = response2.metrics.accumulated_usage

    print()
    print(f"  Turn 2 totals:")
    print(f"    Cache read:  {usage2.get('cacheReadInputTokens', 0):,}")
    print(f"    Cache write: {usage2.get('cacheWriteInputTokens', 0):,}")
    print(f"    Regular:     {usage2.get('inputTokens', 0):,}")

    print()
    print("=" * 60)
    print("What this shows:")
    print()
    print("  Within each turn (tool chaining):")
    print("    - API Call 1: Orchestrator decides → system+tools CACHED")
    print("    - API Call 2-3: Sub-agent runs → SEPARATE cache (different prompt)")
    print("    - API Call 4: Orchestrator continues → system+tools CACHE READ again")
    print()
    print("  Across turns:")
    print("    - System prompt + tools: always CACHE READ (same hash)")
    print("    - Conversation history: grows, computed at regular price")
    print("    - Sub-agents: independent cache, separate from orchestrator")


if __name__ == "__main__":
    main()
