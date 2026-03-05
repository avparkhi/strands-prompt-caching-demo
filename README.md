# Prompt Caching with Strands Multi-Agent on AWS Bedrock

A hands-on demo showing how **prompt caching** works with Claude models on AWS Bedrock, using the Strands Agents framework with a multi-agent (agents-as-tools) architecture.

## What This Project Demonstrates

- **3 caching approaches**: explicit breakpoints, automatic caching, and combined (recommended)
- **Multi-agent orchestration**: an orchestrator routes requests to research and code sub-agents
- **Real cost savings**: see cache metrics and dollar savings on every turn
- **Standalone examples**: runnable scripts that prove how each approach works

## Project Structure

```
Claude_Prompt_Caching/
├── README.md                         # This file
├── BLOG.md                           # Deep-dive: explicit vs automatic vs combined caching
├── requirements.txt                  # Dependencies
├── main.py                           # Multi-agent orchestrator (combined caching)
├── examples/
│   ├── 01_hash_basics.py             # How hash-based cache lookup works
│   ├── 02_cache_write_read.py        # First call writes, second reads
│   ├── 03_two_prompts.py             # Two different prompts = two independent caches
│   ├── 04_multi_turn.py              # Cache savings grow across conversation turns
│   ├── 05_agent_loop.py              # Caching in tool-chaining agent loops
│   ├── 06_explicit_vs_automatic.py   # Side-by-side comparison of all 3 approaches
│   └── 07_databricks_caching.py      # All 3 approaches on Databricks (OpenAI-compatible)
├── agents/
│   ├── __init__.py
│   ├── research_agent.py             # Research sub-agent tool
│   └── code_agent.py                 # Code assistant sub-agent tool
```

## Quick Start

```bash
# 1. Create virtual environment
python3.13 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure AWS credentials (needs Bedrock access)
aws configure

# 4. Run the multi-agent demo (combined caching)
python main.py

# 5. Run individual examples
python examples/01_hash_basics.py
python examples/02_cache_write_read.py
python examples/03_two_prompts.py
python examples/04_multi_turn.py
python examples/05_agent_loop.py
python examples/06_explicit_vs_automatic.py
python examples/07_databricks_caching.py          # requires Databricks credentials
```

## Three Caching Approaches

### 1. Explicit Cache Breakpoints

You manually place `cachePoint` markers on system prompt and tool definitions. Cached from Turn 1, shared across all users.

```python
system_content = [
    SystemContentBlock(text="<system prompt>"),
    SystemContentBlock(cachePoint={"type": "default"}),  # explicit breakpoint
]
model = BedrockModel(model_id="...", cache_tools="default")  # cache tools too
```

### 2. Automatic Caching

Strands auto-injects a `cachePoint` on the last assistant message. No manual markers needed. Kicks in from Turn 2.

```python
from strands.models.model import CacheConfig
model = BedrockModel(model_id="...", cache_config=CacheConfig(strategy="auto"))
```

### 3. Combined (Recommended)

Both explicit (system + tools) and automatic (conversation history). Maximum savings.

```python
system_content = [
    SystemContentBlock(text="<system prompt>"),
    SystemContentBlock(cachePoint={"type": "default"}),
]
model = BedrockModel(
    model_id="us.anthropic.claude-sonnet-4-6",
    cache_tools="default",
    cache_config=CacheConfig(strategy="auto"),
)
```

This is what `main.py` uses. See [BLOG.md](BLOG.md) for the full comparison with pricing data.

## Cache Metrics Example Output

```
Turn 1:
  Cache read:  0 tokens
  Cache write: 2,509 tokens        <-- system prompt + tools stored
  Regular in:  325 tokens
  Cost: $0.010384 (saved -22.1%)   <-- write premium on first turn

Turn 2:
  Cache read:  2,509 tokens        <-- cache hit! 90% cheaper
  Cache write: 2,098 tokens        <-- conversation history stored (automatic)
  Regular in:  0 tokens
  Cost: $0.008620 (saved 86.0%)

Turn 3:
  Cache read:  4,607 tokens        <-- more cached content
  Cache write: 1,773 tokens
  Regular in:  0 tokens
  Cost: $0.008027 (saved 88.6%)    <-- savings grow each turn
```

## Pricing (Claude Sonnet 4.6 on Bedrock)

| Token Type | Price / 1M Tokens | vs Base |
|---|---|---|
| Regular input | $3.00 | 1x |
| Cache write | $3.75 | 1.25x |
| Cache read | $0.30 | 0.1x (90% cheaper) |

## Prerequisites

- Python 3.10+
- AWS account with Bedrock access enabled for Claude Sonnet 4.6
- AWS credentials configured (`aws configure` or environment variables)
