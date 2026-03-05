# Prompt Caching with Strands Multi-Agent on AWS Bedrock & Databricks

A hands-on demo showing how **prompt caching** works with Claude models on **AWS Bedrock** and **Databricks**, using the Strands Agents framework with a multi-agent (agents-as-tools) architecture.

## What This Project Demonstrates

- **3 caching approaches**: explicit breakpoints, automatic caching, and combined (recommended)
- **2 platforms**: AWS Bedrock (native support) and Databricks (via OpenAI-compatible API)
- **Multi-agent orchestration**: an orchestrator routes requests to research and code sub-agents
- **Real cost savings**: see cache metrics and dollar savings on every turn
- **Standalone examples**: runnable scripts that prove how each approach works

## Project Structure

```
Claude_Prompt_Caching/
├── README.md                         # This file
├── requirements.txt                  # Dependencies
├── main.py                           # Multi-agent orchestrator — Bedrock (combined caching)
├── examples/
│   ├── 01_hash_basics.py             # How hash-based cache lookup works
│   ├── 02_cache_write_read.py        # First call writes, second reads
│   ├── 03_two_prompts.py             # Two different prompts = two independent caches
│   ├── 04_multi_turn.py              # Cache savings grow across conversation turns
│   ├── 05_agent_loop.py              # Caching in tool-chaining agent loops
│   ├── 06_explicit_vs_automatic.py   # Side-by-side comparison of all 3 approaches (Bedrock)
│   └── 07_databricks_caching.py      # All 3 approaches on Databricks (Strands OpenAIModel)
├── agents/
│   ├── __init__.py
│   ├── research_agent.py             # Research sub-agent tool
│   └── code_agent.py                 # Code assistant sub-agent tool
```

## Quick Start — AWS Bedrock

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
```

## Quick Start — Databricks

```bash
# 1. Install OpenAI dependency (in addition to requirements.txt)
pip install openai

# 2. Set Databricks credentials
export DATABRICKS_HOST="https://<workspace>.cloud.databricks.com"
export DATABRICKS_TOKEN="dapi..."
export DATABRICKS_MODEL_ENDPOINT="databricks-claude-sonnet-4-6"

# 3. Run with any caching approach
python examples/07_databricks_caching.py combined    # recommended
python examples/07_databricks_caching.py explicit
python examples/07_databricks_caching.py automatic
```

## Three Caching Approaches

### 1. Explicit Cache Breakpoints

You manually place cache markers on system prompt and tool definitions. Cached from Turn 1, shared across all users.

**Bedrock** — uses `cachePoint`:
```python
system_content = [
    SystemContentBlock(text="<system prompt>"),
    SystemContentBlock(cachePoint={"type": "default"}),
]
model = BedrockModel(model_id="...", cache_tools="default")
```

**Databricks** — uses `cache_control` (injected via `CachedOpenAIModel`):
```python
# cache_control: {"type": "ephemeral"} added to system message + last tool definition
model = CachedOpenAIModel(cache_approach="explicit", ...)
```

### 2. Automatic Caching

Cache point auto-injected on the last assistant message. No manual markers needed. Kicks in from Turn 2.

**Bedrock** — built-in:
```python
from strands.models.model import CacheConfig
model = BedrockModel(model_id="...", cache_config=CacheConfig(strategy="auto"))
```

**Databricks** — via `CachedOpenAIModel`:
```python
# cache_control injected on last assistant message before each API call
model = CachedOpenAIModel(cache_approach="automatic", ...)
```

### 3. Combined (Recommended)

Both explicit (system + tools) and automatic (conversation history). Maximum savings.

**Bedrock**:
```python
model = BedrockModel(
    model_id="us.anthropic.claude-sonnet-4-6",
    cache_tools="default",
    cache_config=CacheConfig(strategy="auto"),
)
```

**Databricks**:
```python
model = CachedOpenAIModel(cache_approach="combined", ...)
```

## Real Test Results — Databricks

3-turn conversation with Claude Sonnet 4.6 on Databricks:

### Explicit
| Turn | Cache Read | Regular | Savings |
|---|---|---|---|
| 1 | 4,992 | 772 | **77.9%** |
| 2 | 9,984 | 1,878 | **75.8%** |
| 3 | 14,976 | 3,710 | **72.1%** |

### Automatic
| Turn | Cache Read | Regular | Savings |
|---|---|---|---|
| 1 | 0 | 5,764 | **0.0%** |
| 2 | 2,979 | 8,883 | **22.6%** |
| 3 | 9,269 | 9,549 | **44.2%** |

### Combined
| Turn | Cache Read | Regular | Savings |
|---|---|---|---|
| 1 | 4,992 | 772 | **77.9%** |
| 2 | 9,984 | 1,395 | **74.7%** |
| 3 | 16,108 | 1,850 | **76.8%** |

**Key takeaways:**
- **Explicit** saves from Turn 1 but conversation history stays at full price (regular tokens grow)
- **Automatic** wastes Turn 1 entirely, slowly catches up
- **Combined** gets the best of both — high savings from Turn 1, regular tokens stay low

## Bedrock vs Databricks — How Caching Differs

| Feature | AWS Bedrock | Databricks |
|---|---|---|
| API format | Bedrock Converse API | OpenAI-compatible |
| Cache syntax | `cachePoint: {"type": "default"}` | `cache_control: {"type": "ephemeral"}` |
| Strands support | Native (`BedrockModel`) | Via `CachedOpenAIModel` subclass (workaround) |
| Explicit caching | `cache_tools="default"` + `SystemContentBlock(cachePoint=...)` | Injected in `format_request()` |
| Automatic caching | `CacheConfig(strategy="auto")` | Injected in `format_request()` |
| Cache metrics | `cacheReadInputTokens` / `cacheWriteInputTokens` | `cache_read_input_tokens` / `cache_creation_input_tokens` |

> **Note:** Strands' `OpenAIModel` doesn't natively support caching yet ([issue #1140](https://github.com/strands-agents/sdk-python/issues/1140)). The `CachedOpenAIModel` in `examples/07_databricks_caching.py` works around this by overriding `format_request()` to inject `cache_control` markers and `format_chunk()` to pass through cache metrics.

## Pricing (Claude Sonnet 4.6)

| Token Type | Price / 1M Tokens | vs Base |
|---|---|---|
| Regular input | $3.00 | 1x |
| Cache write | $3.75 | 1.25x |
| Cache read | $0.30 | 0.1x (90% cheaper) |

## Prerequisites

- Python 3.10+
- **For Bedrock**: AWS account with Bedrock access for Claude Sonnet 4.6 + `aws configure`
- **For Databricks**: Workspace with Claude model serving endpoint + `DATABRICKS_HOST` and `DATABRICKS_TOKEN` env vars
