"""MCP Server for context-optimiser.

Exposes tools to estimate token usage, compact context, rotate sessions, and query vector memory.
"""
import sys
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from mcp.server.fastmcp import FastMCP
from swarm.context_compactor import ContextCompactor
from swarm.chat_rotator import ChatRotator
from swarm.vector_store import VectorMemoryStore
import context_rotation_config as config

# Initialize FastMCP Server
mcp = FastMCP("Context Optimiser MCP Server")

@mcp.tool()
def estimate_tokens(conversation: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Estimate the token size of a conversation using the context-optimiser rules."""
    compactor = ContextCompactor(token_limit=config.TOKEN_LIMIT)
    stats = compactor.estimate_tokens(conversation)
    return {
        "tokens_used": stats.tokens_used,
        "limit": stats.limit,
        "pct_used": stats.pct_used,
        "can_rotate": stats.can_rotate
    }

@mcp.tool()
def compact_context(
    conversation: List[Dict[str, Any]], 
    strategy: str = "combined", 
    keep_last_n_turns: int = 4
) -> List[Dict[str, Any]]:
    """Compacts the conversation by clustering failure attempts and offloading completed tasks."""
    compactor = ContextCompactor(token_limit=config.TOKEN_LIMIT)
    return compactor.compact(conversation, strategy=strategy, keep_last_n_turns=keep_last_n_turns)

@mcp.tool()
def rotate_session(
    conversation: List[Dict[str, Any]],
    origin_prompt: str,
    token_limit: int = 16000,
    threshold_pct: int = 85
) -> Dict[str, Any]:
    """Checks token usage, compacts context and rotates the chat session if token limit is exceeded."""
    compactor = ContextCompactor(token_limit=token_limit)
    stats = compactor.estimate_tokens(conversation)
    
    if stats.pct_used < threshold_pct:
        return {
            "rotated": False,
            "tokens_used": stats.tokens_used,
            "pct_used": stats.pct_used,
            "compacted_conversation": conversation
        }
    
    # Try compaction first
    compacted = compactor.compact(conversation, keep_last_n_turns=4)
    compacted_stats = compactor.estimate_tokens(compacted)
    
    # Check if compaction succeeded in bringing it below the threshold
    # If AUTO_ROTATE_AFTER_COMPACTION is True (default) and compacted size is below threshold, skip rotation!
    auto_rotate_after_compactor = getattr(config, 'AUTO_ROTATE_AFTER_COMPACTION', True)
    if auto_rotate_after_compactor and compacted_stats.pct_used < threshold_pct:
        return {
            "rotated": False,
            "tokens_used": compacted_stats.tokens_used,
            "pct_used": compacted_stats.pct_used,
            "compacted_conversation": compacted
        }
        
    rotator = ChatRotator()
    session_file = rotator.find_session_file()
    
    # Load current session, or construct a default one if none exists
    if session_file and session_file.exists():
        current_session = rotator.load_session(session_file) or {}
    else:
        current_session = {
            "session_id": "root",
            "phase": "active"
        }
        
    # Get compaction details
    reduction_pct = 0.0
    initial_tokens = stats.tokens_used
    if initial_tokens > 0:
        reduction_pct = ((initial_tokens - compacted_stats.tokens_used) / initial_tokens) * 100
        
    # Create new session
    new_session = rotator.create_new_session(
        current_session=current_session,
        compaction_stats={"reduction_pct": reduction_pct},
        origin_prompt=origin_prompt
    )
    
    # Write new session file
    new_session_path = rotator.write_new_session(new_session)
    
    # Update link on current active session if it existed
    if session_file and session_file.exists():
        rotator.update_current_session_link(session_file, new_session["session_id"])
        
    # Log the rotation
    link_info = rotator.create_chain_link(
        current_session=current_session,
        new_session=new_session,
        compaction_reduction_pct=reduction_pct,
        origin_prompt=origin_prompt
    )
    rotator.log_rotation(link_info, new_session_path)
    
    # Overwrite/Update the main session.json file to point to the new session
    # so subsequent tools know the active session.
    try:
        rotator.session_file.write_text(json.dumps(new_session, indent=2), encoding="utf-8")
    except Exception:
        pass
        
    notification = rotator.get_notification_message(link_info)
    
    return {
        "rotated": True,
        "new_session_id": new_session["session_id"],
        "notification": notification,
        "compacted_conversation": compacted
    }

@mcp.tool()
def get_current_session() -> Dict[str, Any]:
    """Retrieve details of the current active session."""
    rotator = ChatRotator()
    session_file = rotator.find_session_file()
    if session_file and session_file.exists():
        session = rotator.load_session(session_file)
        if session:
            return session
    return {"status": "no_active_session"}

@mcp.tool()
def query_vector_store(query_text: str, n_results: int = 3) -> List[Dict[str, Any]]:
    """Query offloaded contexts saved in ChromaDB."""
    vdb = VectorMemoryStore()
    return vdb.query(query_text=query_text, n_results=n_results)

if __name__ == "__main__":
    mcp.run()
