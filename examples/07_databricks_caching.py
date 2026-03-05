"""
Example 7: Prompt Caching on Databricks with Claude using Strands OpenAIModel
==============================================================================

Uses Strands' OpenAIModel provider pointed at Databricks' OpenAI-compatible
endpoint serving Claude. Demonstrates all three caching approaches with the
agents-as-tools pattern (same as main.py).

Strands' OpenAIModel does NOT yet support prompt caching natively (there's an
open issue: https://github.com/strands-agents/sdk-python/issues/1140).

This example works around that by subclassing OpenAIModel to:
  1. Inject cache_control markers into the formatted request (format_request)
  2. Pass through cache token metrics from the response (format_chunk)

Approaches:
  Explicit  — cache_control on system prompt + tool definitions
  Automatic — cache_control injected on last assistant message each turn
  Combined  — Explicit + Automatic together (recommended)

Requires:
  - pip install 'strands-agents[openai]'
  - Databricks workspace with a Claude model serving endpoint
  - DATABRICKS_HOST and DATABRICKS_TOKEN environment variables

Usage:
  python examples/07_databricks_caching.py                  # default: combined
  python examples/07_databricks_caching.py explicit
  python examples/07_databricks_caching.py automatic
  python examples/07_databricks_caching.py combined
"""

import os
import sys
import re
import copy
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strands import Agent, tool
from strands.models.openai import OpenAIModel
from strands.types.streaming import StreamEvent
from strands.agent.conversation_manager.sliding_window_conversation_manager import (
    SlidingWindowConversationManager,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST", "https://<workspace>.databricks.com")
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
MODEL_ENDPOINT = os.environ.get("DATABRICKS_MODEL_ENDPOINT", "databricks-claude-sonnet")

# Pricing (Sonnet 4.6)
INPUT_PRICE = 3.00
CACHE_READ_PRICE = 0.30
CACHE_WRITE_PRICE = 3.75
OUTPUT_PRICE = 15.00

# Load system prompt from main.py
main_py_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py")
with open(main_py_path) as f:
    _content = f.read()
_match = re.search(r'ORCHESTRATOR_SYSTEM_PROMPT = """(.+?)"""', _content, re.DOTALL)
SYSTEM_PROMPT = _match.group(1)


# =========================================================================
# CachedOpenAIModel — adds prompt caching to Strands' OpenAIModel
# =========================================================================
# Strands' OpenAIModel._format_system_messages() currently drops cachePoint
# blocks (line 329 in openai.py: "TODO: Handle caching blocks #1140").
# And format_chunk() doesn't pass through cache token metrics.
#
# This subclass fixes both by:
# 1. Overriding format_request() to inject cache_control into the formatted
#    request dict (system messages, tools, last assistant message)
# 2. Overriding format_chunk() to include cache metrics in the usage dict
# =========================================================================

class CachedOpenAIModel(OpenAIModel):
    """OpenAIModel with prompt caching support for Databricks/Anthropic endpoints."""

    def __init__(self, cache_approach: str = "combined", **kwargs: Any):
        """
        Args:
            cache_approach: One of "explicit", "automatic", "combined".
            **kwargs: Passed to OpenAIModel.__init__.
        """
        super().__init__(**kwargs)
        self.cache_approach = cache_approach

    def format_request(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Format request then inject cache_control markers based on approach."""
        request = super().format_request(*args, **kwargs)

        messages = copy.deepcopy(request.get("messages", []))
        tools = copy.deepcopy(request.get("tools", []))

        # --- Explicit: cache system prompt ---
        if self.cache_approach in ("explicit", "combined"):
            for msg in messages:
                if msg.get("role") == "system":
                    content = msg.get("content")
                    if isinstance(content, str):
                        # Convert to content block format so we can add cache_control
                        msg["content"] = [
                            {
                                "type": "text",
                                "text": content,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ]
                    elif isinstance(content, list) and len(content) > 0:
                        content[-1]["cache_control"] = {"type": "ephemeral"}
                    break

        # --- Explicit: cache tool definitions ---
        if self.cache_approach in ("explicit", "combined"):
            if tools:
                tools[-1]["cache_control"] = {"type": "ephemeral"}

        # --- Automatic: cache last assistant message ---
        if self.cache_approach in ("automatic", "combined"):
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") != "assistant":
                    continue
                content = messages[i].get("content")
                if isinstance(content, str):
                    messages[i]["content"] = [
                        {
                            "type": "text",
                            "text": content,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ]
                elif isinstance(content, list) and len(content) > 0:
                    content[-1]["cache_control"] = {"type": "ephemeral"}
                break

        request["messages"] = messages
        if tools:
            request["tools"] = tools

        return request

    def format_chunk(self, event: dict[str, Any], **kwargs: Any) -> StreamEvent:
        """Format chunk, adding cache token metrics to usage if present."""
        if event.get("chunk_type") == "metadata":
            usage_data = event.get("data")
            base_usage: dict[str, Any] = {
                "inputTokens": getattr(usage_data, "prompt_tokens", 0),
                "outputTokens": getattr(usage_data, "completion_tokens", 0),
                "totalTokens": getattr(usage_data, "total_tokens", 0),
            }

            # Databricks returns cache metrics as extra fields on the usage object
            # when serving Anthropic models with cache_control
            cache_read = getattr(usage_data, "cache_read_input_tokens", 0) or 0
            cache_write = getattr(usage_data, "cache_creation_input_tokens", 0) or 0

            if cache_read:
                base_usage["cacheReadInputTokens"] = cache_read
            if cache_write:
                base_usage["cacheWriteInputTokens"] = cache_write

            return {
                "metadata": {
                    "usage": base_usage,
                    "metrics": {"latencyMs": 0},
                },
            }

        return super().format_chunk(event, **kwargs)


# =========================================================================
# Sub-agents as tools (agents-as-tools pattern, same as main.py)
# =========================================================================

def _make_model(cache_approach: str = "combined") -> CachedOpenAIModel:
    return CachedOpenAIModel(
        cache_approach=cache_approach,
        client_args={
            "api_key": DATABRICKS_TOKEN,
            "base_url": f"{DATABRICKS_HOST}/serving-endpoints",
        },
        model_id=MODEL_ENDPOINT,
        params={"max_tokens": 1024},
    )


@tool
def research_assistant(query: str) -> str:
    """Research factual information and provide well-sourced answers."""
    agent = Agent(
        model=_make_model(),
        system_prompt="You are a research assistant. Provide clear, accurate, well-sourced answers.",
    )
    return str(agent(query))


@tool
def code_assistant(query: str) -> str:
    """Write, review, and debug code."""
    agent = Agent(
        model=_make_model(),
        system_prompt="You are a code assistant. Write clean, well-documented, production-ready code.",
    )
    return str(agent(query))


# =========================================================================
# Cache metrics display
# =========================================================================

def print_cache_metrics(response, turn):
    """Print cache performance metrics and cost savings."""
    usage = response.metrics.accumulated_usage

    cache_read = usage.get("cacheReadInputTokens", 0)
    cache_write = usage.get("cacheWriteInputTokens", 0)
    input_tokens = usage.get("inputTokens", 0)
    output_tokens = usage.get("outputTokens", 0)

    # With cache metrics, regular = total input minus cached portions
    if cache_read or cache_write:
        regular = max(0, input_tokens - cache_read - cache_write)
    else:
        regular = input_tokens

    cost_no_cache = (cache_read + cache_write + regular) * INPUT_PRICE / 1_000_000
    cost_with_cache = (
        cache_read * CACHE_READ_PRICE / 1_000_000
        + cache_write * CACHE_WRITE_PRICE / 1_000_000
        + regular * INPUT_PRICE / 1_000_000
    )
    output_cost = output_tokens * OUTPUT_PRICE / 1_000_000
    total_cost = cost_with_cache + output_cost
    savings = (1 - cost_with_cache / cost_no_cache) * 100 if cost_no_cache > 0 else 0

    print(f"\n  --- Cache Metrics (Turn {turn}) ---")
    print(f"  Cache read:  {cache_read:>6,} tokens  (${cache_read * CACHE_READ_PRICE / 1_000_000:.6f})")
    print(f"  Cache write: {cache_write:>6,} tokens  (${cache_write * CACHE_WRITE_PRICE / 1_000_000:.6f})")
    print(f"  Regular in:  {regular:>6,} tokens  (${regular * INPUT_PRICE / 1_000_000:.6f})")
    print(f"  Output:      {output_tokens:>6,} tokens  (${output_cost:.6f})")
    print(f"  Total cost:  ${total_cost:.6f}  (saved {savings:.1f}% on input vs no cache)")
    print(f"  --------------------\n")


# =========================================================================
# Main
# =========================================================================

def main():
    approach = sys.argv[1] if len(sys.argv) > 1 else "combined"
    if approach not in ("explicit", "automatic", "combined"):
        print(f"Usage: python {sys.argv[0]} [explicit|automatic|combined]")
        sys.exit(1)

    model = CachedOpenAIModel(
        cache_approach=approach,
        client_args={
            "api_key": DATABRICKS_TOKEN,
            "base_url": f"{DATABRICKS_HOST}/serving-endpoints",
        },
        model_id=MODEL_ENDPOINT,
        params={"max_tokens": 1024},
    )

    orchestrator = Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[research_assistant, code_assistant],
        conversation_manager=SlidingWindowConversationManager(window_size=20),
    )

    print("=" * 60)
    print(f"Databricks + Strands Caching — {approach.upper()}")
    print("=" * 60)
    print()
    print(f"Caching approach: {approach}")
    if approach == "explicit":
        print("  cache_control on: system prompt + tool definitions")
        print("  Conversation history: NOT cached (regular price)")
    elif approach == "automatic":
        print("  cache_control on: last assistant message (auto-injected)")
        print("  System prompt + tools: NOT cached independently")
    else:
        print("  cache_control on: system prompt + tools (explicit)")
        print("  cache_control on: last assistant message (automatic)")
        print("  Maximum savings from Turn 1 onward")
    print()
    print("Type 'quit' to exit.\n")

    turn = 0
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        turn += 1
        print(f"\n[Turn {turn}]")

        response = orchestrator(user_input)
        print(f"\nAssistant: {response}")
        print_cache_metrics(response, turn)


if __name__ == "__main__":
    main()
