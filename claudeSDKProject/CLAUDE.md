# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the agent

```bash
source .venv/bin/activate
python research_agent.py
```

## Architecture

This is a single-file stateless research agent using the `claude-agent-sdk` Python package.

- **`research_agent.py`** — the entire agent. Uses `query()` from `claude_agent_sdk` for one-off, stateless interactions. Each call to `query()` is independent with no shared session or memory between queries.
- **Pattern**: `async for message in query(prompt=..., options=ClaudeAgentOptions(system_prompt=...))` — iterates over streamed message objects, checking `message.content` blocks for `block.text` to print responses.
- The agent is focused on health/nutrition research (protein digestion), but the `run_research_query` helper is generic and reusable for any topic by swapping the system prompt and query list in `main()`.
