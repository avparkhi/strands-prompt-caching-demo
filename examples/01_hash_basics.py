"""
Example 1: How Hash-Based Cache Lookup Works
=============================================

No API calls needed. This shows the pure math behind cache key generation.

A hash function takes any input and produces a fixed-size fingerprint.
Same input → same hash, always. Different input → different hash, always.

This is how Anthropic identifies cache hits: it hashes the bytes before each
cachePoint marker and uses that hash as the cache key. No comparison against
stored prompts, no session management — just math.
"""

import hashlib


def compute_hash(text: str) -> str:
    """Compute SHA-256 hash of text, just like Anthropic does with prompt bytes."""
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    print("=" * 60)
    print("Example 1: How Hash-Based Cache Lookup Works")
    print("=" * 60)

    # --- Same text produces the same hash ---
    print("\n--- Same text → same hash ---\n")

    prompt_a = "You are a great developer"
    prompt_b = "You are a great developer"

    hash_a = compute_hash(prompt_a)
    hash_b = compute_hash(prompt_b)

    print(f'Prompt A: "{prompt_a}"')
    print(f"Hash A:   {hash_a}")
    print()
    print(f'Prompt B: "{prompt_b}"  (identical)')
    print(f"Hash B:   {hash_b}")
    print(f"Match:    {hash_a == hash_b}")

    # --- Even a tiny change produces a completely different hash ---
    print("\n--- Tiny change → completely different hash ---\n")

    prompt_c = "You are a great developer."  # added a period

    hash_c = compute_hash(prompt_c)

    print(f'Prompt C: "{prompt_c}"  (added one dot)')
    print(f"Hash C:   {hash_c}")
    print(f"Match A:  {hash_a == hash_c}")

    # --- Different prompts, different hashes ---
    print("\n--- Two different system prompts → two independent cache keys ---\n")

    support_prompt = "You are a customer support agent for Acme Corp."
    code_prompt = "You are a senior code reviewer specializing in Python."

    hash_support = compute_hash(support_prompt)
    hash_code = compute_hash(code_prompt)

    print(f'Support prompt hash: {hash_support[:32]}...')
    print(f'Code prompt hash:    {hash_code[:32]}...')
    print(f"Match: {hash_support == hash_code}")

    # --- Simulate Anthropic's cache lookup ---
    print("\n--- Simulating Anthropic's cache store ---\n")

    cache_store = {}

    prompts_to_process = [
        ("User 1", support_prompt),
        ("User 2", code_prompt),
        ("User 3", support_prompt),  # same as User 1
        ("User 4", code_prompt),     # same as User 2
        ("User 5", support_prompt),  # same as User 1
    ]

    for user, prompt in prompts_to_process:
        key = compute_hash(prompt)[:16]
        if key in cache_store:
            print(f"  {user}: hash={key}  CACHE HIT  (same as {cache_store[key]})")
        else:
            cache_store[key] = user
            print(f"  {user}: hash={key}  CACHE MISS → stored")

    print()
    print("Key takeaway: The hash is computed from the text every time.")
    print("Same text → same hash → cache hit. No session needed.")


if __name__ == "__main__":
    main()
