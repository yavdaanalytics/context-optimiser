---
name: context-optimiser
description: |
  Exposes tools and routines to optimize conversation context, rotate chats, offload
  completed tasks to ChromaDB, and cluster semantic failures.
license: Apache-2.0
metadata:
  version: v1
  publisher: yavdaanalytics
---

# Context Optimiser Loader & Management Skill

This skill allows the agent to monitor context usage and invoke the `context-optimiser` MCP server to compact, offload, or rotate conversations when the context window fills up or when repetitive failures cluster.

## 1. Auto-Loading & Registering the MCP Server

Before using any of the context-optimiser tools, verify that the `context-optimiser` MCP server is registered in your active editor/platform configurations. If it is missing, dynamically edit/create the configuration files to register it:

### Target Configuration Files
- **Gemini / Antigravity**: `~/.gemini/settings.json`
- **Cursor**: `~/.cursor/mcp.json`
- **Claude Desktop**: Claude desktop MCP config (platform-specific path)

### Registration Format (published package)

```json
{
  "mcpServers": {
    "context-optimiser": {
      "command": "npx",
      "args": ["-y", "@yavdaanalytics/context-optimiser"]
    }
  }
}
```

Or run `npx -y -p @yavdaanalytics/context-optimiser context-optimiser-setup`.

For a local checkout, use `node bin/setup.js --local` (Python path to `mcp_server.py`).

---

## 2. Using the MCP Tools

Once the MCP server is registered and loaded, you can call the following tools to manage context:

### A. estimate_tokens
- **Purpose**: Heuristically estimate the token count of the current conversation.
- **Parameters**: `conversation` (List of message dicts)
- **Usage**: Call before deciding whether to trigger compaction or rotation.

### B. compact_context
- **Purpose**: Compact a conversation sequence using semantic failure clustering (TAG) and heuristic offloading.
- **Parameters**:
  - `conversation` (List of message dicts)
  - `strategy` (default: `"combined"`)
  - `keep_last_n_turns` (default: `4`)

### C. rotate_session
- **Purpose**: Rotate the conversation session once token usage exceeds a specified threshold (default: 85% of limit).
- **Parameters**:
  - `conversation` (List of message dicts)
  - `origin_prompt` (String that triggered rotation)
  - `token_limit` (default: `16000`)
  - `threshold_pct` (default: `85`)
- **Behavior**: If threshold is exceeded, this will compact the context, create a new linked session ID, and return the compacted conversation along with rotation stats.

### D. query_vector_store
- **Purpose**: Retrieve historical offloaded conversations using semantic search from the local ChromaDB.
- **Parameters**:
  - `query_text` (Search query string)
  - `n_results` (default: `3`)
