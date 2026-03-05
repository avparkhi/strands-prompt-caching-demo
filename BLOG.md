# Prompt Caching on AWS Bedrock with Claude: Explicit, Automatic, and Combined

*A deep dive into the three caching approaches available on AWS Bedrock, with hands-on examples using Strands Agents.*

---

## Table of Contents

1. [What Is Prompt Caching?](#1-what-is-prompt-caching)
2. [How It Works Under the Hood](#2-how-it-works-under-the-hood)
3. [The Three Caching Approaches](#3-the-three-caching-approaches)
4. [Approach 1: Explicit Cache Breakpoints](#4-approach-1-explicit-cache-breakpoints)
5. [Approach 2: Automatic Caching](#5-approach-2-automatic-caching)
6. [Approach 3: Combined (Explicit + Automatic)](#6-approach-3-combined-explicit--automatic)
7. [Pricing Comparison: Which Approach Wins?](#7-pricing-comparison-which-approach-wins)
8. [How Caching Works in Agent Tool Loops](#8-how-caching-works-in-agent-tool-loops)
9. [Cross-User Sharing and Multiple Prompts](#9-cross-user-sharing-and-multiple-prompts)
10. [Important Gotchas and Minimum Requirements](#10-important-gotchas-and-minimum-requirements)
11. [Running the Examples](#11-running-the-examples)

---

## 1. What Is Prompt Caching?

Prompt caching saves the **processed internal state** (KV attention tensors) of a prompt prefix so it can be reused across API requests, avoiding redundant GPU computation.

```
Without caching:
  Request 1: [System Prompt: 2000 tokens] + [User msg: 50 tokens]  → Compute ALL 2050 tokens
  Request 2: [System Prompt: 2000 tokens] + [User msg: 80 tokens]  → Compute ALL 2080 tokens
               ^^^^^^^^^^^^^^^^^^^^^^^^ Redundant! Recomputed from scratch.

With caching:
  Request 1: [System Prompt: 2000 tokens] + [User msg: 50 tokens]  → WRITE 2000 to cache + compute 50
  Request 2: [System Prompt: 2000 tokens] + [User msg: 80 tokens]  → READ 2000 from cache + compute 80
```

**Result**: Cache reads are **90% cheaper** than regular input tokens and reduce time-to-first-token.

---

## 2. How It Works Under the Hood

### Hash-Based Lookup

Anthropic uses a hash of the bytes before each `cachePoint` marker to identify cache entries. No session management, no scanning — pure math.

```
Step 1: Your request arrives with a cachePoint marker
Step 2: Server computes hash of all bytes before the marker  (~0.001ms)
Step 3: Hash table lookup — O(1)
          cache["a3f8b2c1..."] exists?
            YES → Cache HIT → load precomputed KV tensors (READ)
            NO  → Cache MISS → compute KV tensors, store them (WRITE)
Step 4: Process the rest of the request normally
```

The hash is **computed**, not assigned. Same bytes always produce the same hash. Different bytes (even one extra space) produce a different hash.

```python
import hashlib
hashlib.sha256("You are a great developer".encode()).hexdigest()
# → "709af3a..." — always the same for the same input
hashlib.sha256("You are a great developer.".encode()).hexdigest()
# → "ac835e0..." — completely different (one dot added!)
```

> Run `python examples/01_hash_basics.py` to see this yourself.

### The KV Cache

When a transformer processes tokens, each layer computes Key and Value tensors for attention. These KV tensors are what gets cached — they represent the fully processed state of those tokens, so loading them skips all GPU computation for the cached prefix.

### Cache Lifecycle

```
Time 0:00  → First request   → Cache MISS → WRITE    TTL starts: 5 min
Time 1:00  → Same prefix     → Cache HIT  → READ     TTL resets: 5 min
Time 3:00  → Same prefix     → Cache HIT  → READ     TTL resets: 5 min
Time 8:00  → No hits for 5m  → EVICTED
Time 8:01  → Same prefix     → Cache MISS → WRITE    TTL starts: 5 min
```

Every hit resets the 5-minute TTL. High-traffic apps keep the cache warm naturally.

> Run `python examples/02_cache_write_read.py` to see a WRITE then READ in action.

---

## 3. The Three Caching Approaches

AWS Bedrock with Strands Agents supports three approaches to prompt caching:

| Approach | What It Caches | When It Starts Saving | Who Benefits |
|---|---|---|---|
| **Explicit** | System prompt + tool definitions | Turn 1 (write), Turn 1+ (reads within agent loop) | All users (shared prefix) |
| **Automatic** | Conversation history | Turn 2+ (auto-injects cachePoint on last assistant msg) | Per-user (their own history) |
| **Combined** | All of the above | Turn 1 for system+tools, Turn 2+ for history | Maximum savings |

### Max Cache Breakpoints

You can have at most **4 cache breakpoints** per request. The combined approach typically uses:
1. System prompt cachePoint (explicit)
2. Tool definitions cachePoint (explicit via `cache_tools`)
3. Last assistant message cachePoint (automatic via `CacheConfig`)

---

## 4. Approach 1: Explicit Cache Breakpoints

You manually place `cachePoint` markers at fixed positions in the request. These are cached from the very first API call.

### What You Cache

1. **System prompt** — via `SystemContentBlock(cachePoint={"type": "default"})`
2. **Tool definitions** — via `cache_tools="default"` on the model

### Setup with Strands Agents

```python
from strands import Agent
from strands.models import BedrockModel
from strands.types.content import SystemContentBlock

# 1. System prompt with explicit cache breakpoint
system_content = [
    SystemContentBlock(text="<your long system prompt — must exceed 2048 tokens for Sonnet 4.6>"),
    SystemContentBlock(cachePoint={"type": "default"}),  # cache everything above
]

# 2. Model with tool caching
model = BedrockModel(
    model_id="us.anthropic.claude-sonnet-4-6",
    cache_tools="default",  # adds cachePoint after tool definitions
)

# 3. Agent
agent = Agent(
    model=model,
    system_prompt=system_content,
    tools=[your_tool_1, your_tool_2],
)
```

### Setup with Raw Boto3

```python
import boto3

client = boto3.client("bedrock-runtime", region_name="us-east-1")

response = client.converse_stream(
    modelId="us.anthropic.claude-sonnet-4-6",
    system=[
        {"text": "<your long system prompt>"},
        {"cachePoint": {"type": "default"}},   # explicit breakpoint
    ],
    toolConfig={
        "tools": [...],
        "cachePoint": {"type": "default"},     # cache tool definitions
    },
    messages=[{"role": "user", "content": [{"text": "Hello"}]}],
    inferenceConfig={"maxTokens": 500},
)
```

### How It Behaves

```
Turn 1, API Call 1 (agent decides):
  [System Prompt WRITE][Tools WRITE][User msg]
  → System + tools written to cache (1.25x cost)

Turn 1, API Call 2 (after tool result, same turn):
  [System Prompt READ][Tools READ][Messages]
  → System + tools already cached! 90% cheaper on these tokens

Turn 2:
  [System Prompt READ][Tools READ][Turn 1 history + User msg 2]
  → System + tools cached, but conversation history is at full price

Turn 3:
  [System Prompt READ][Tools READ][Turn 1-2 history + User msg 3]
  → Same pattern — history keeps growing at full price
```

**Strengths**: System prompt and tools cached immediately (from first API call). Shared across all users.

**Weakness**: Conversation history is never cached — it's always computed at full price.

### What Strands Does Under the Hood

When you set `cache_tools="default"`, Strands adds a `cachePoint` after the tool definitions in the Bedrock Converse API `toolConfig`. The `SystemContentBlock(cachePoint=...)` is passed directly in the `system` parameter.

> Run `python examples/04_multi_turn.py` for a multi-turn demo with explicit caching.

---

## 5. Approach 2: Automatic Caching

Strands automatically injects a `cachePoint` on the **last assistant message** in conversation history. The cache point moves forward each turn, so progressively more of the conversation gets cached.

### Setup with Strands Agents

```python
from strands import Agent
from strands.models import BedrockModel
from strands.models.model import CacheConfig

model = BedrockModel(
    model_id="us.anthropic.claude-sonnet-4-6",
    cache_config=CacheConfig(strategy="auto"),  # automatic conversation caching
)

agent = Agent(
    model=model,
    system_prompt="<your system prompt>",
    tools=[your_tool_1, your_tool_2],
)
```

Note: No explicit `cachePoint` on the system prompt, no `cache_tools`. Only the automatic strategy.

### How It Behaves

```
Turn 1:
  [System Prompt][Tools][User msg 1]
  → No cachePoint injected (no prior assistant message yet)
  → Nothing cached! Regular pricing on everything.

Turn 2:
  [System Prompt][Tools][User msg 1][Asst reply 1 ← cachePoint injected HERE][User msg 2]
  → Everything before cachePoint: WRITE (system + tools + turn 1)
  → User msg 2: regular price

Turn 3:
  [System Prompt][Tools][Turn 1][Asst reply 2 ← cachePoint moved HERE][User msg 3]
  → System + tools + turns 1-2: READ from cache
  → User msg 3: regular price
```

**Strengths**: Zero configuration. History gets progressively cheaper. Cache point automatically moves forward.

**Weakness**: Nothing cached on Turn 1 (no prior assistant message). System prompt and tools are only cached as part of the conversation prefix starting Turn 2 — not independently cached on Turn 1.

### What Strands Does Under the Hood

In `strands/models/bedrock.py`, the `_inject_cache_point()` method finds the last assistant message in the conversation and adds a `cachePoint` marker after it. This happens in `_format_request()` when `cache_config.strategy == "auto"`.

---

## 6. Approach 3: Combined (Explicit + Automatic)

The best of both worlds. Use explicit breakpoints for system prompt and tools (cached from Turn 1), plus automatic caching for conversation history (cached from Turn 2+).

### Setup with Strands Agents

```python
from strands import Agent
from strands.models import BedrockModel
from strands.models.model import CacheConfig
from strands.types.content import SystemContentBlock

# Explicit: cache system prompt
system_content = [
    SystemContentBlock(text="<your long system prompt>"),
    SystemContentBlock(cachePoint={"type": "default"}),
]

# Explicit (tools) + Automatic (conversation)
model = BedrockModel(
    model_id="us.anthropic.claude-sonnet-4-6",
    cache_tools="default",                       # explicit: cache tool definitions
    cache_config=CacheConfig(strategy="auto"),   # automatic: cache conversation history
)

agent = Agent(
    model=model,
    system_prompt=system_content,
    tools=[your_tool_1, your_tool_2],
)
```

### How It Behaves

```
Turn 1, API Call 1:
  [System Prompt WRITE][Tools WRITE][User msg 1]
  → System + tools cached immediately (explicit)
  → No conversation history yet

Turn 1, API Call 2 (after tool result):
  [System Prompt READ][Tools READ][Messages]
  → System + tools already cached from API Call 1!

Turn 2:
  [System Prompt READ][Tools READ][Turn 1 history ← auto cachePoint][User msg 2]
  → System + tools: READ (explicit, 90% cheaper)
  → Turn 1 history: READ or WRITE (automatic)
  → User msg 2: regular price

Turn 3:
  [System Prompt READ][Tools READ][Turns 1-2 ← auto cachePoint moves][User msg 3]
  → System + tools: READ
  → Turns 1-2: READ (automatic, growing savings)
  → User msg 3: regular price
```

**Strengths**: Maximum savings. System+tools cached from Turn 1 (shared across all users). Conversation history cached from Turn 2+ (per-user). The auto cachePoint moves forward each turn.

**Weakness**: Uses 3 of 4 available cache breakpoints (system, tools, auto conversation).

This is what `main.py` in this project uses.

---

## 7. Pricing Comparison: Which Approach Wins?

### Bedrock Pricing (Claude Sonnet 4.6)

| Token Type | Price per 1M Tokens | Relative to Base |
|---|---|---|
| Regular input | $3.00 | 1x |
| Cache write | $3.75 | 1.25x (25% premium) |
| Cache read | $0.30 | 0.1x (90% discount) |
| Output | $15.00 | — |

### Real Test Data: 4-Turn Conversation

Using a ~2,500 token system prompt with two tool-using sub-agents:

#### Explicit Only

| Turn | Cache Read | Cache Write | Regular | Input Cost | Savings vs No Cache |
|---|---|---|---|---|---|
| 1 | 0 | 2,509 | 325 | $0.010384 | -22.1% |
| 2 | 2,509 | 0 | 2,098 | $0.007047 | 83.4% |
| 3 | 2,509 | 0 | 3,871 | $0.012366 | 88.1% |
| 4 | 2,509 | 0 | 5,644 | $0.017685 | 88.1% |

#### Automatic Only

| Turn | Cache Read | Cache Write | Regular | Input Cost | Savings vs No Cache |
|---|---|---|---|---|---|
| 1 | 0 | 0 | 2,834 | $0.008502 | 0.0% |
| 2 | 0 | 2,834 | 2,098 | $0.016923 | -21.7% |
| 3 | 2,834 | 2,098 | 1,037 | $0.011814 | 67.4% |
| 4 | 4,932 | 3,135 | 1,037 | $0.016260 | 67.4% |

#### Combined (Explicit + Automatic)

| Turn | Cache Read | Cache Write | Regular | Input Cost | Savings vs No Cache |
|---|---|---|---|---|---|
| 1 | 0 | 2,509 | 325 | $0.010384 | -22.1% |
| 2 | 2,509 | 2,098 | 0 | $0.008620 | 86.0% |
| 3 | 4,607 | 1,773 | 0 | $0.008027 | 88.6% |
| 4 | 6,380 | 1,773 | 0 | $0.008559 | 88.6% |

### Summary

| Approach | Turn 1 | Turn 3 Savings | Turn 4 Savings | Best For |
|---|---|---|---|---|
| **Explicit** | -22.1% (write premium) | 88.1% | 88.1% | Short conversations, shared system prompt |
| **Automatic** | 0% (nothing cached) | 67.4% | 67.4% | Simple setup, long conversations |
| **Combined** | -22.1% (write premium) | 88.6% | 88.6% | Maximum savings, production use |

**Key takeaways:**
- **Explicit** starts saving within the same turn (agent loop reuses system+tools) but never caches conversation history
- **Automatic** wastes Turn 1 entirely (no prior assistant message), catches up from Turn 2
- **Combined** gets the best of both — immediate system+tool caching plus growing conversation savings
- The write premium on Turn 1 pays for itself immediately in multi-turn or multi-user scenarios

---

## 8. How Caching Works in Agent Tool Loops

When an agent uses tools, it makes **multiple API calls per user turn**:

```
User: "Explain Python and write hello world"

API Call 1: Orchestrator decides what to do
  [System Prompt][cachePoint][Tools][cachePoint][User msg]
  → System + tools: WRITE (or READ if already cached)
  Response: tool_use: research_assistant("Explain Python")

    Sub-agent runs (SEPARATE cache — different system prompt, different hash)
    API Call 2: research_assistant processes query

API Call 3: Orchestrator continues
  [System Prompt][cachePoint][Tools][cachePoint][User msg + tool_use + tool_result]
  → System + tools: CACHE READ (same hash as API Call 1!)
  → Messages after tools: regular price (growing with tool results)
```

### Sub-Agents Have Independent Caches

Each sub-agent has its own system prompt, so its own hash and cache entry:

```
Orchestrator:        hash("You are an orchestrator...") → "abc123"
research_assistant:  hash("You are a research...")      → "def456"
code_assistant:      hash("You are a code...")          → "ghi789"
```

Three independent caches. The orchestrator's cache is hit on every API call in its loop. Sub-agent caches are hit when the same sub-agent is called again.

> Run `python examples/05_agent_loop.py` to see per-API-call cache metrics during tool chaining.

---

## 9. Cross-User Sharing and Multiple Prompts

### Same Prompt, Different Users = Shared Cache

```
User 1:   [System Prompt 2500 tok][cachePoint] + [Chat about Python]
User 2:   [System Prompt 2500 tok][cachePoint] + [Chat about AWS]
User 100: [System Prompt 2500 tok][cachePoint] + [Chat about databases]

           IDENTICAL prefix → CACHED ONCE          ALL DIFFERENT → computed fresh
           Shared by all 100 users
```

The cache is per-prefix-hash, not per-user or per-API-key. Anyone sending the same bytes gets the same cache entry.

### Different Prompts = Independent Caches

```
App A:  system_prompt = "You are a support agent..."  → hash "abc123"
App B:  system_prompt = "You are a code reviewer..."  → hash "xyz789"
```

Two completely independent cache entries. They never interfere, even on the same API key.

> Run `python examples/03_two_prompts.py` to see independent caches side by side.

---

## 10. Important Gotchas and Minimum Requirements

### Minimum Token Thresholds

The prefix before a `cachePoint` must exceed a minimum for caching to activate:

| Model | Minimum Tokens |
|---|---|
| Claude Opus 4.6, Opus 4.5 | 4,096 |
| Claude Sonnet 4.6 | 2,048 |
| Claude Sonnet 4.5, 4, Opus 4.1, 4 | 1,024 |
| Claude Haiku 4.5 | 4,096 |
| Claude Haiku 3.5, 3 | 2,048 |

If your system prompt is too short, caching silently does nothing — you'll see `cacheReadInputTokens: 0` and `cacheWriteInputTokens: 0` in every response. No error, no warning.

**Solution**: Make your system prompt detailed enough (routing rules, examples, guidelines, expertise maps) to exceed the threshold.

### Exact Byte Matching

The prefix must be **byte-for-byte identical**:

```
"You are a helpful assistant"   → hash: abc123
"You are a helpful assistant "  → hash: def456  (trailing space!)
"You are a helpful assistant."  → hash: ghi789  (added period!)
```

**Avoid** dynamic content (timestamps, request IDs, user names) before the cache point.

### Max 4 Cache Breakpoints Per Request

You can place at most 4 `cachePoint` markers in a single request. The combined approach uses 3 (system, tools, conversation).

### Cache TTL: 5 Minutes

- Default TTL is ~5 minutes from last hit
- Every cache READ resets the timer
- Optional: request 1-hour TTL at 2x base input price (not commonly used)
- No way to manually inspect, extend, or invalidate cache entries

### Streaming Required on Bedrock

The Bedrock `Converse` (non-streaming) API does not return cache token metrics. Use `ConverseStream` to see `cacheReadInputTokens` and `cacheWriteInputTokens`. Strands uses streaming by default, so this works out of the box.

### Anthropic Direct API vs Bedrock Syntax

```python
# Anthropic Direct API:
{"type": "text", "text": "...", "cache_control": {"type": "ephemeral"}}

# AWS Bedrock Converse API:
[{"text": "..."}, {"cachePoint": {"type": "default"}}]
```

Same concept, different syntax. Strands abstracts this — you use `SystemContentBlock` either way.

---

## 11. Running the Examples

### Prerequisites

```bash
# Python 3.10+
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# AWS credentials with Bedrock access for Claude Sonnet 4.6
aws configure
```

### Example Scripts

```bash
# 1. How hash-based lookup works (no API calls needed)
python examples/01_hash_basics.py

# 2. First call writes cache, second reads it
python examples/02_cache_write_read.py

# 3. Two different prompts = two independent caches
python examples/03_two_prompts.py

# 4. Multi-turn conversation with growing cache savings
python examples/04_multi_turn.py

# 5. Per-API-call metrics during agent tool chaining
python examples/05_agent_loop.py

# 6. Side-by-side comparison of explicit vs automatic vs combined
python examples/06_explicit_vs_automatic.py

# Full interactive multi-agent demo (combined caching)
python main.py
```

### What to Look For

**Example 02** — First call shows `CACHE WRITE`, second shows `CACHE READ`. Same prompt, same hash.

**Example 04** — Watch `Cache read` tokens grow each turn while system prompt stays cached:
```
Turn 1: Cache read:      0  |  Cache write:  2,509  |  Regular:    50
Turn 2: Cache read:  2,509  |  Cache write:      0  |  Regular:   200
Turn 3: Cache read:  2,509  |  Cache write:      0  |  Regular:   500
Turn 4: Cache read:  2,509  |  Cache write:      0  |  Regular:   800
```

**Example 06** — Compare all three approaches head-to-head with real pricing.

**main.py** — Interactive chat using the combined approach. Each turn prints cache metrics and cost savings.

---

## Summary

| Question | Answer |
|---|---|
| What gets cached? | KV attention tensors (processed state of tokens) |
| Where is the cache? | Anthropic/AWS server-side GPU memory |
| How is a cache hit identified? | Hash of bytes before `cachePoint` — same bytes = same hash |
| Is there session management? | No. Pure hash lookup. No sessions, no affinity |
| How long does cache last? | ~5 min TTL, resets on each hit |
| Do different prompts interfere? | No. Different text = different hash = independent cache |
| Is cache shared across users? | Yes, if they send the same prefix bytes |
| Which approach should I use? | **Combined** for production. Explicit for simple cases. Automatic for zero-config. |
| What's the minimum prefix size? | 2,048 tokens for Sonnet 4.6 on Bedrock |
| How much does it save? | Up to 90% on cached input tokens (88.6% overall in our tests) |

---

*Built with [Strands Agents](https://github.com/strands-agents/strands-agents) and AWS Bedrock.*
