# @yavdaanalytics/context-optimiser

A context window optimizer and session rotator MCP server for agentic workflows. It prevents LLMs/coding agents from running out of context window space by estimating token usage, compacting chat history using semantic failure clustering (Topic Attempt Graph - TAG), and rotating chat sessions dynamically while preserving session continuity links.

## Features

1. **Token Estimation**: Heuristically counts and budgets tokens for a conversation.
2. **Context Compaction**: Uses semantic failure clustering (TAG) and heuristic offloading to shrink conversational history, offloading large code dumps/logs to disk or local ChromaDB.
3. **Session Auto-Rotation**: Dynamically rotates chat sessions when context limit thresholds are reached, updating next/previous session pointers for linked context history.
4. **Local Vector Search**: Stores offloaded messages in ChromaDB and retrieves them via semantic similarity search.

---

## Quick Start

### 1. Installation

Install globally via npm:
```bash
npm install -g @yavdaanalytics/context-optimiser
```

### 2. Setup

Run the setup utility to configure the MCP server globally for Gemini, Cursor, and Claude Desktop, and copy the agent loading skills:
```bash
context-optimiser-setup
```

This will automatically configure:
- **Gemini**: `~/.gemini/settings.json`
- **Cursor**: `~/.cursor/mcp.json`
- **Claude Desktop**: `~/.claude/settings.json` and `%APPDATA%/Claude/claude_desktop_config.json`
- **Loader Skills**: Copies `SKILL.md` to `~/.cursor/skills/context-optimiser/SKILL.md`, `~/.claude/skills/context-optimiser/SKILL.md`, and `~/.gemini/config/skills/context-optimiser/SKILL.md`.

---

## MCP Tools API

The server registers the following MCP tools for client use:

- `estimate_tokens(conversation)`: Returns estimated token usage, capacity limits, and percentage used.
- `compact_context(conversation, strategy, keep_last_n_turns)`: Returns compacted message history.
- `rotate_session(conversation, origin_prompt, token_limit, threshold_pct)`: Returns rotated session metadata and compacted history.
- `get_current_session()`: Returns active session JSON metadata.
- `query_vector_store(query_text, n_results)`: Performs semantic vector similarity queries on offloaded history.

## Development & Local Testing

If you are developing locally, run setup in local mode to link the configuration to your checkout directory:
```bash
node bin/setup.js --local
```

### Run Tests
To verify all tool integrations and ChromaDB vector queries:
```bash
python scratch/test_tools.py
```
