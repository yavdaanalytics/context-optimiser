"""Stub GPT5AdaptiveCompactor for static analysis and safe runtime fallback.

This stub provides a minimal class with the same public constructor signature so
the executor can import it without failing static checks. It implements a 
safe sliding-window compaction fallback for production robustness.
"""
from typing import Any


class GPT5AdaptiveCompactor:
    def __init__(self, model: str = "gpt-5-mini", token_limit: int = 16000, compaction_threshold_pct: int = 85, enable_extended_thinking: bool = False):
        self.model = model
        self.token_limit = token_limit
        self.compaction_threshold_pct = compaction_threshold_pct
        self.enable_extended_thinking = enable_extended_thinking

    def compact(self, conversation: Any, **kwargs) -> Any:
        """
        Performs a simple sliding-window compaction to prevent context overflow.
        Keeps the system message (if any) and the most recent N messages.
        """
        if not isinstance(conversation, list):
            return conversation
            
        if len(conversation) <= 6:
            return conversation
            
        # Keep system message + last 5 messages
        compacted = []
        if conversation and conversation[0].get("role") == "system":
            compacted.append(conversation[0])
            
        compacted.extend(conversation[-5:])
        return compacted
