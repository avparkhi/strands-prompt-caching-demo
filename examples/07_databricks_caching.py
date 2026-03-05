"""
Example 7: Prompt Caching on Databricks with Claude
====================================================

Databricks serves Claude via an OpenAI-compatible API with Anthropic's
cache_control extension. This example shows all three caching approaches:

  Approach 1 — Explicit:  cache_control on system prompt + tool definitions
  Approach 2 — Automatic: cache_control injected on last assistant message
  Approach 3 — Combined:  Explicit + Automatic together (recommended)

Requires:
  - Databricks workspace with a Claude model serving endpoint
  - pip install openai  (or: pip install databricks-openai)
  - DATABRICKS_HOST and DATABRICKS_TOKEN environment variables

Usage:
  python examples/07_databricks_caching.py                  # default: combined
  python examples/07_databricks_caching.py explicit
  python examples/07_databricks_caching.py automatic
  python examples/07_databricks_caching.py combined
"""

import os
import sys
import copy

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — update these for your Databricks workspace
# ---------------------------------------------------------------------------
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST", "https://<workspace>.databricks.com")
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
MODEL_ENDPOINT = os.environ.get("DATABRICKS_MODEL_ENDPOINT", "databricks-claude-sonnet")

# Pricing (Sonnet 4.6 — adjust if using a different model)
INPUT_PRICE = 3.00
CACHE_READ_PRICE = 0.30
CACHE_WRITE_PRICE = 3.75
OUTPUT_PRICE = 15.00

# System prompt — must exceed minimum token threshold (~2048 for Sonnet 4.6)
SYSTEM_PROMPT = """You are a senior AI orchestrator that routes user requests to specialized sub-agents.

## Your Role
You coordinate between specialized assistants to provide the best possible response.
You should analyze each user request and determine which assistant(s) to invoke.

## Available Assistants

### Research Assistant
Use for: factual questions, explanations, comparisons, summaries, historical context,
scientific topics, current events analysis, and any request requiring well-sourced information.

### Code Assistant
Use for: writing code, debugging errors, code review, refactoring suggestions,
explaining code snippets, architecture advice, and any programming-related request.

## Routing Guidelines

1. Single-domain requests: Route to the appropriate assistant directly.
2. Mixed requests: Break down into parts and route each to the right assistant.
3. Ambiguous requests: Default to research for information, code for programming.
4. Follow-up questions: Consider conversation context to determine routing.

## Response Guidelines

- Synthesize responses from sub-agents into a coherent answer
- Add your own context or transitions when combining multiple sub-agent responses
- If a sub-agent response seems incomplete, call it again with a refined query
- Keep the overall response focused and well-structured

## Quality Standards

### Accuracy and Completeness
- Verify that responses address the core question
- Ensure code examples are complete and runnable
- Cross-reference information between agents when both contribute

### Response Structure
- Use clear headings and subheadings for long responses
- Present code blocks with appropriate language tags
- Use bullet points and numbered lists for parallel information
- Include summary sections for complex multi-part responses

### Error Handling
- If a sub-agent fails, attempt once more with a simplified query
- For ambiguous queries, ask a clarifying question
- Never present incorrect information as fact

### Context Management
- Maintain awareness of the full conversation history
- Reference previous responses when relevant
- Avoid redundant calls for information already provided

## Domain-Specific Routing

### Technical Documentation
- Research for concepts, Code for implementation
- Use both for comprehensive guides

### Debugging
- Code for syntax/runtime errors
- Research for conceptual misunderstandings
- Both when bug stems from concept misunderstanding

### Learning Requests
- Research for theory, Code for hands-on examples
- Present theory before practice

## Programming Language Expertise

### Python
- Web: Django, Flask, FastAPI
- Data: pandas, numpy, scikit-learn
- Async: asyncio, aiohttp
- Testing: pytest, unittest

### JavaScript/TypeScript
- Frontend: React, Vue, Angular, Next.js
- Backend: Node.js, Express, NestJS
- Testing: Jest, Vitest, Playwright

### Cloud and Infrastructure
- AWS: Lambda, EC2, S3, DynamoDB, Bedrock
- Docker and Kubernetes
- Terraform, CI/CD pipelines

### AI and Machine Learning
- PyTorch, TensorFlow, Hugging Face
- LLM APIs: Anthropic, OpenAI, Bedrock
- Agent frameworks: Strands, LangChain
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "research_assistant",
            "description": "Research factual information and provide well-sourced answers.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The research query"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code_assistant",
            "description": "Write, review, and debug code.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The coding query"}},
                "required": ["query"],
            },
        },
    },
]


# =========================================================================
# Cache injection helpers
# =========================================================================

def _add_cache_control(content_block):
    """Add cache_control to a content block (dict)."""
    block = copy.deepcopy(content_block)
    block["cache_control"] = {"type": "ephemeral"}
    return block


def _cache_system(system_prompt):
    """Return system message WITH cache_control on text block."""
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
    }


def _plain_system(system_prompt):
    """Return system message WITHOUT cache_control."""
    return {
        "role": "system",
        "content": [{"type": "text", "text": system_prompt}],
    }


def _cache_tools(tools):
    """Add cache_control to the last tool definition."""
    if not tools:
        return tools
    cached = copy.deepcopy(tools)
    cached[-1]["cache_control"] = {"type": "ephemeral"}
    return cached


def inject_auto_cache(messages):
    """
    Find the last assistant message and add cache_control to its last content block.

    This replicates what Strands does with CacheConfig(strategy="auto") on Bedrock:
    it injects a cachePoint on the last assistant message so that everything before it
    (system prompt + tools + conversation history up to that point) gets cached.
    The cache point moves forward each turn as new assistant messages are added.
    """
    messages = [copy.copy(msg) for msg in messages]

    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] != "assistant":
            continue

        content = messages[i].get("content")

        if isinstance(content, str):
            messages[i] = {
                **messages[i],
                "content": [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        elif isinstance(content, list) and len(content) > 0:
            content = [copy.deepcopy(b) for b in content]
            content[-1]["cache_control"] = {"type": "ephemeral"}
            messages[i] = {**messages[i], "content": content}
        break

    return messages


# =========================================================================
# Three caching approaches
# =========================================================================

def build_explicit(system_prompt, tools, conversation):
    """Approach 1: Explicit cache_control on system prompt + tool definitions."""
    return _cache_system(system_prompt), _cache_tools(tools), list(conversation)


def build_automatic(system_prompt, tools, conversation):
    """Approach 2: Auto-inject cache_control on last assistant message only."""
    return _plain_system(system_prompt), tools, inject_auto_cache(conversation)


def build_combined(system_prompt, tools, conversation):
    """Approach 3: Explicit (system+tools) + Automatic (conversation history)."""
    return _cache_system(system_prompt), _cache_tools(tools), inject_auto_cache(conversation)


BUILDERS = {
    "explicit": build_explicit,
    "automatic": build_automatic,
    "combined": build_combined,
}


# =========================================================================
# Chat loop
# =========================================================================

def print_usage(usage, turn):
    """Print cache metrics if available."""
    # Databricks may return these fields in the usage object
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0

    # Regular input = total prompt minus cached portions
    regular = max(0, prompt_tokens - cache_read - cache_write)

    cost_no_cache = prompt_tokens * INPUT_PRICE / 1_000_000
    cost_with_cache = (
        cache_read * CACHE_READ_PRICE / 1_000_000
        + cache_write * CACHE_WRITE_PRICE / 1_000_000
        + regular * INPUT_PRICE / 1_000_000
    )
    output_cost = completion_tokens * OUTPUT_PRICE / 1_000_000
    savings = (1 - cost_with_cache / cost_no_cache) * 100 if cost_no_cache > 0 else 0

    print(f"\n  --- Cache Metrics (Turn {turn}) ---")
    print(f"  Cache read:  {cache_read:>6,} tokens  (${cache_read * CACHE_READ_PRICE / 1_000_000:.6f})")
    print(f"  Cache write: {cache_write:>6,} tokens  (${cache_write * CACHE_WRITE_PRICE / 1_000_000:.6f})")
    print(f"  Regular in:  {regular:>6,} tokens  (${regular * INPUT_PRICE / 1_000_000:.6f})")
    print(f"  Output:      {completion_tokens:>6,} tokens  (${output_cost:.6f})")
    print(f"  Savings:     {savings:.1f}% on input vs no cache")
    print(f"  --------------------\n")


def main():
    approach = sys.argv[1] if len(sys.argv) > 1 else "combined"
    if approach not in BUILDERS:
        print(f"Usage: python {sys.argv[0]} [explicit|automatic|combined]")
        sys.exit(1)

    builder = BUILDERS[approach]

    client = OpenAI(
        api_key=DATABRICKS_TOKEN,
        base_url=f"{DATABRICKS_HOST}/serving-endpoints",
    )

    print("=" * 60)
    print(f"Databricks Claude Caching — Approach: {approach.upper()}")
    print("=" * 60)
    print()
    print("Type 'quit' to exit.\n")

    conversation = []
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
        conversation.append({"role": "user", "content": user_input})

        # Build messages with chosen caching approach
        system_msg, tools, cached_conversation = builder(
            SYSTEM_PROMPT, TOOLS, conversation
        )

        kwargs = {
            "model": MODEL_ENDPOINT,
            "messages": [system_msg] + cached_conversation,
            "max_tokens": 1024,
        }
        if tools:
            kwargs["tools"] = tools

        print(f"\n[Turn {turn}]")
        response = client.chat.completions.create(**kwargs)

        assistant_msg = response.choices[0].message.content or ""
        conversation.append({"role": "assistant", "content": assistant_msg})

        print(f"\nAssistant: {assistant_msg}")
        print_usage(response.usage, turn)


if __name__ == "__main__":
    main()
