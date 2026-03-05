"""
Example 2: Cache Write on First Call, Read on Second
=====================================================

Makes two identical API calls to AWS Bedrock with a cachePoint marker.
- Call 1: Cache MISS → computes KV tensors → stores in cache (WRITE)
- Call 2: Cache HIT → loads KV tensors from cache (READ, 90% cheaper)

This proves that caching is server-side and automatic once you add the marker.
No session ID, no cache key management — just the same bytes = same cache.

Requires: AWS credentials with Bedrock access
"""

import boto3
import json

client = boto3.client("bedrock-runtime", region_name="us-east-1")

# System prompt must exceed ~2048 tokens for caching on Sonnet 4.6
SYSTEM_PROMPT = (
    "You are a helpful AI assistant. "
    + "Provide detailed, accurate, and well-structured responses to all queries. " * 300
)


def call_bedrock(label: str):
    """Make a streaming Bedrock API call and print cache metrics."""
    response = client.converse_stream(
        modelId="us.anthropic.claude-sonnet-4-6",
        system=[
            {"text": SYSTEM_PROMPT},
            {"cachePoint": {"type": "default"}},  # ← THIS enables caching
        ],
        messages=[{"content": [{"text": "Say hello in one word"}], "role": "user"}],
        inferenceConfig={"maxTokens": 10},
    )

    for event in response["stream"]:
        if "metadata" in event:
            usage = event["metadata"].get("usage", {})
            write = usage.get("cacheWriteInputTokens", 0)
            read = usage.get("cacheReadInputTokens", 0)
            regular = usage.get("inputTokens", 0)

            if write > 0:
                status = "CACHE WRITE (new entry stored)"
            elif read > 0:
                status = "CACHE READ  (loaded from cache!)"
            else:
                status = "NO CACHE    (prompt too short?)"

            print(f"  {label}:")
            print(f"    Cache write:  {write:,} tokens")
            print(f"    Cache read:   {read:,} tokens")
            print(f"    Regular input: {regular:,} tokens")
            print(f"    Status: {status}")
            print()


def main():
    print("=" * 60)
    print("Example 2: Cache Write on First Call, Read on Second")
    print("=" * 60)
    print()

    print("Call 1: First request with this system prompt")
    call_bedrock("Call 1")

    print("Call 2: Identical request — should hit cache")
    call_bedrock("Call 2")

    print("What happened:")
    print("  Call 1: Anthropic computed hash of system prompt bytes.")
    print("          Hash not found in cache → computed KV tensors → stored (WRITE).")
    print()
    print("  Call 2: Same system prompt → same hash.")
    print("          Hash found in cache → loaded KV tensors → skipped computation (READ).")
    print("          90% cheaper on those tokens!")
    print()
    print("  No session, no API key check, no connection affinity.")
    print("  Just: same bytes → same hash → cache hit.")


if __name__ == "__main__":
    main()
