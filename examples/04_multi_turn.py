"""
Example 4: Multi-Turn Conversation with Growing Cache Savings
==============================================================

Shows how prompt caching works across multiple conversation turns.
The system prompt + tools are cached from the first call and reused every turn.
Conversation history grows, but the cached prefix stays the same.

Turn 1: Cache WRITE (system prompt stored)
Turn 2: Cache READ on system prompt + growing history
Turn 3: Cache READ — even more tokens cached
Turn 4: Cache READ — savings keep growing

Requires: AWS credentials with Bedrock access
"""

from strands import Agent
from strands.models import BedrockModel
from strands.types.content import SystemContentBlock
from strands.agent.conversation_manager.sliding_window_conversation_manager import (
    SlidingWindowConversationManager,
)

# System prompt large enough for caching (>2048 tokens on Sonnet 4.6)
SYSTEM_PROMPT = """You are a helpful AI assistant specializing in technology topics.

Provide clear, concise, and accurate answers. Structure your responses with:
1. A direct answer to the question
2. Supporting details and context
3. Examples when helpful

""" + (
    "Always maintain a professional and helpful tone. "
    "Consider the user's level of expertise when explaining concepts. "
    "Use analogies to make complex topics accessible. "
    "Cite best practices from industry leaders when relevant. "
) * 100

system_content = [
    SystemContentBlock(text=SYSTEM_PROMPT),
    SystemContentBlock(cachePoint={"type": "default"}),
]

model = BedrockModel(
    model_id="us.anthropic.claude-sonnet-4-6",
    cache_tools="default",
)

agent = Agent(
    model=model,
    system_prompt=system_content,
    conversation_manager=SlidingWindowConversationManager(window_size=20),
)

# Simulated multi-turn conversation
QUESTIONS = [
    "What is Python? Answer in 2 sentences.",
    "What about JavaScript? 2 sentences.",
    "Compare them briefly.",
    "Which should I learn first? One sentence.",
]

# Sonnet pricing
INPUT_PRICE = 3.00
CACHE_READ_PRICE = 0.30
CACHE_WRITE_PRICE = 3.75


def main():
    print("=" * 60)
    print("Example 4: Multi-Turn Conversation with Cache Savings")
    print("=" * 60)
    print()

    cumulative_saved = 0.0
    cumulative_cost = 0.0

    for turn_num, question in enumerate(QUESTIONS, 1):
        print(f"Turn {turn_num}: \"{question}\"")

        response = agent(question)
        usage = response.metrics.accumulated_usage

        cache_read = usage.get("cacheReadInputTokens", 0)
        cache_write = usage.get("cacheWriteInputTokens", 0)
        input_tokens = usage.get("inputTokens", 0)

        # Cost calculation
        cost_no_cache = (cache_read + cache_write + input_tokens) * INPUT_PRICE / 1_000_000
        cost_with_cache = (
            cache_read * CACHE_READ_PRICE / 1_000_000
            + cache_write * CACHE_WRITE_PRICE / 1_000_000
            + input_tokens * INPUT_PRICE / 1_000_000
        )
        saved = cost_no_cache - cost_with_cache
        savings_pct = (saved / cost_no_cache * 100) if cost_no_cache > 0 else 0

        cumulative_saved += saved
        cumulative_cost += cost_with_cache

        print(f"  Cache read: {cache_read:>6,}  |  Cache write: {cache_write:>6,}  |  Regular: {input_tokens:>6,}")
        print(f"  Input cost: ${cost_with_cache:.6f}  (saved {savings_pct:.1f}% vs no cache)")
        print()

    print("-" * 60)
    print(f"Total cost with caching:    ${cumulative_cost:.6f}")
    print(f"Total saved vs no caching:  ${cumulative_saved:.6f}")
    print()
    print("Notice how cache reads grow each turn:")
    print("  Turn 1: System prompt cached (write or read depending on warm cache)")
    print("  Turn 2: System prompt + Turn 1 history = more cache reads")
    print("  Turn 3: System prompt + Turn 1-2 history = even more")
    print("  Turn 4: System prompt + Turn 1-3 history = maximum savings")


if __name__ == "__main__":
    main()
