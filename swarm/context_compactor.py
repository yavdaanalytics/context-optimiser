"""Heuristic and Semantic context compactor used as a fallback in executor.

Implements Topic Attempt Graph (TAG) for semantic failure clustering and
Intent Preservation for efficient context handling.
"""
import json
import logging
import sys
from pathlib import Path
from typing import Any, List, Dict
from .context_offloader import compact_conversation_with_pointers

# Import Adaptive Compactor
try:
    sys.path.append(str(Path(__file__).resolve().parent.parent / ".agent" / "scripts"))
    from gpt5_adaptive_compactor import GPT5AdaptiveCompactor
except Exception:
    GPT5AdaptiveCompactor = None

log = logging.getLogger("swarm.compactor")

class ContextCompactor:
    def __init__(self, token_limit: int = 16000):
        self.token_limit = token_limit
        self.adaptive_compactor = GPT5AdaptiveCompactor() if GPT5AdaptiveCompactor else None

    class Stats:
        def __init__(self, used: int, limit: int):
            self.tokens_used = used
            self.limit = limit
            self.pct_used = (used / limit) * 100
            self.can_rotate = True

    def estimate_tokens(self, conversation: List[Dict]) -> Stats:
        # Rough heuristic estimation: ~4 chars per token
        chars = sum(len(str(m.get("content", ""))) for m in conversation)
        return self.Stats(chars // 4, self.token_limit)

    def compact(self, conversation: List[Dict], strategy: str = "combined", keep_last_n_turns: int = 4) -> List[Dict]:
        """
        Compacts the conversation by clustering failed attempts into summary blocks (TAG)
        and offloading completed tasks using semantic pointers.
        """
        # 1. Semantic Failure Clustering (Topic Attempt Graph - TAG)
        if self.adaptive_compactor:
            conversation = self._cluster_failures(conversation)

        # 2. Heuristic Offloading (Intent Preservation)
        return compact_conversation_with_pointers(conversation, keep_last_n=keep_last_n_turns)

    def _cluster_failures(self, conversation: List[Dict]) -> List[Dict]:
        """
        Identifies repeated failed attempts at the same topic and collapses them.
        """
        if len(conversation) < 10:
            return conversation

        # Heuristic: Find sequences of messages that look like repeated attempts
        # (e.g. multiple tool failures followed by similar user re-prompts)
        clustered = []
        i = 0
        while i < len(conversation):
            msg = conversation[i]
            content = str(msg.get("content", ""))
            
            # If we see a "failure" or "error" in a message, look ahead for repetitions
            if "error" in content.lower() or "failed" in content.lower():
                # Potential failure sequence
                sequence = [msg]
                j = i + 1
                while j < len(conversation) and j < i + 6:
                    next_msg = conversation[j]
                    next_content = str(next_msg.get("content", ""))
                    if "error" in next_content.lower() or "failed" in next_content.lower():
                        sequence.append(next_msg)
                        j += 1
                    else:
                        break
                
                if len(sequence) >= 2:
                    # We have a failure cluster! Summarize it.
                    log.info(f"[compactor] TAG: Clustering failure sequence of {len(sequence)} messages")
                    summary = self._summarize_failure_sequence(sequence)
                    clustered.append({
                        "role": "system",
                        "content": f"[[FAILURE_SUMMARY_BLOCK]]\nSummary of failed attempts: {summary}\nNote: The individual failed steps were collapsed to save context."
                    })
                    # Also keep the last message in the sequence as it usually has the most recent error details
                    clustered.append(sequence[-1])
                    i = j
                    continue
            
            clustered.append(msg)
            i += 1
            
        return clustered

    def _summarize_failure_sequence(self, sequence: List[Dict]) -> str:
        """
        Uses GPT-5-Mini to summarize a sequence of failed attempts.
        """
        if not self.adaptive_compactor:
            return "Multiple attempts failed due to recurring errors."
            
        try:
            # Build a minimal prompt for summarization
            prompt = f"Summarize the following sequence of failed interaction attempts into a single concise paragraph. Focus on WHAT was attempted and WHY it failed.\n\nSequence:\n{json.dumps(sequence, indent=2)}"
            
            # Use the adaptive compactor for the actual summarization
            result = self.adaptive_compactor.compact(
                conversation=[{"role": "user", "content": prompt}],
                token_budget=2000
            )
            
            if result.success:
                # Extract the summarized content from the compacted conversation
                # (AdaptiveCompactor returns a list of messages)
                summary_msg = result.conversation[0].get("content", "")
                return summary_msg
        except Exception as e:
            log.warning(f"[compactor] TAG summarization failed: {e}")
            
        return "Multiple attempts failed due to recurring errors."
