"""
Example 3: Two Different Prompts = Two Independent Caches
==========================================================

Proves that different system prompts create completely independent cache entries.
They never interfere with each other, even on the same API key.

Sequence:
  Request 1: Prompt A (support agent)     → CACHE WRITE (new hash)
  Request 2: Prompt B (code reviewer)     → CACHE WRITE (different hash)
  Request 3: Prompt A again               → CACHE READ  (matches Request 1's hash!)
  Request 4: Prompt B again               → CACHE READ  (matches Request 2's hash!)
  Request 5: Prompt A again               → CACHE READ  (still matches!)

Requires: AWS credentials with Bedrock access
"""

import boto3
import json

client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Two different system prompts — each will get its own cache entry
PROMPT_A = (
    "You are a customer support agent for Acme Corporation. "
    + "Handle all customer queries with empathy, professionalism, and detailed solutions. " * 300
)

PROMPT_B = (
    "You are a senior code reviewer specializing in Python and JavaScript. "
    + "Review all code for bugs, security vulnerabilities, and performance improvements. " * 300
)


def call_bedrock(label: str, system_prompt: str):
    """Make a Bedrock call and print cache metrics."""
    response = client.converse_stream(
        modelId="us.anthropic.claude-sonnet-4-6",
        system=[
            {"text": system_prompt},
            {"cachePoint": {"type": "default"}},
        ],
        messages=[{"content": [{"text": "Hi"}], "role": "user"}],
        inferenceConfig={"maxTokens": 5},
    )

    for event in response["stream"]:
        if "metadata" in event:
            usage = event["metadata"].get("usage", {})
            write = usage.get("cacheWriteInputTokens", 0)
            read = usage.get("cacheReadInputTokens", 0)

            if write > 0:
                status = f"CACHE WRITE  ({write:,} tokens stored)"
            elif read > 0:
                status = f"CACHE READ   ({read:,} tokens loaded)"
            else:
                status = "NO CACHE"

            print(f"  {label}: {status}")


def main():
    print("=" * 60)
    print("Example 3: Two Different Prompts = Two Independent Caches")
    print("=" * 60)
    print()

    print("Request 1: Prompt A (support agent)")
    call_bedrock("Result", PROMPT_A)
    print()

    print("Request 2: Prompt B (code reviewer) — DIFFERENT text, DIFFERENT hash")
    call_bedrock("Result", PROMPT_B)
    print()

    print("Request 3: Prompt A again — SAME text as Request 1, SAME hash")
    call_bedrock("Result", PROMPT_A)
    print()

    print("Request 4: Prompt B again — SAME text as Request 2, SAME hash")
    call_bedrock("Result", PROMPT_B)
    print()

    print("Request 5: Prompt A again")
    call_bedrock("Result", PROMPT_A)
    print()

    print("What this proves:")
    print("  - Two different prompts = two independent cache entries")
    print("  - They never interfere (Request 3 hits A's cache, not B's)")
    print("  - Same API key, same endpoint — caching is purely hash-based")
    print("  - You can run 100 different apps on one API key, each cached independently")


if __name__ == "__main__":
    main()
