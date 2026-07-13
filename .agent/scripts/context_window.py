"""
scripts/context_window.py — AutoBeast Context Window Manager (P1 Phantom Module)

Provides token budget enforcement and text truncation helpers.
Previously referenced by executor.py and worker.py but never implemented.

Uses a simple char-count heuristic if tiktoken is not installed,
falling back gracefully to avoid hard dependencies.

Canonical path source: paths.py
"""
from __future__ import annotations
import logging
import re

log = logging.getLogger("context_window")

# Approximate token ratios
_CHARS_PER_TOKEN = 3.8  # Conservative estimate for English + code

# Model context limits (in tokens) — conservative usable limits
_MODEL_LIMITS: dict[str, int] = {
    "claude-sonnet-4-5":  180_000,
    "claude-opus-4":       180_000,
    "claude-haiku-3-5":   180_000,
    "gpt-5-mini":          127_000,
    "gpt-5.4-mini":        127_000,
    "gpt-4o-mini":         127_000,
    "gpt-4o":              127_000,
    "default":              64_000,
}

# Safety margin — keep 15% of context for model output
_OUTPUT_RESERVE = 0.15


def _estimate_tokens(text: str) -> int:
    """Estimate token count from character count. Tries tiktoken first."""
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Fallback: char-count heuristic
        return max(1, int(len(text) / _CHARS_PER_TOKEN))


def _usable_limit(model: str) -> int:
    """Return usable token limit for a model (total - output reserve)."""
    limit = _MODEL_LIMITS.get(model, _MODEL_LIMITS["default"])
    return int(limit * (1.0 - _OUTPUT_RESERVE))


class ContextWindowManager:
    """
    Token budget enforcer and text trimmer.

    Usage (in executor.py / worker.py):
        cw = _load_cw()
        if cw:
            ok, used, limit = cw.budget_check(prompt_text, model)
            if not ok:
                prompt_text = cw.trim_text(prompt_text, model, keep="tail")
    """

    def __init__(self, model: str = "default"):
        self.model = model
        self.limit = _usable_limit(model)

    def count_tokens(self, text: str) -> int:
        """Alias for estimate() — expected by executor.py and worker.py."""
        return _estimate_tokens(text)

    def budget_check(self, text: str, model: str | None = None, label: str = "generic") -> dict:
        """
        Check if text fits within the model's context budget.
        Returns a decision dict expected by swarm/executor.py.
        """
        m = model or self.model
        limit = _usable_limit(m)
        tokens = _estimate_tokens(text)
        pct = (tokens / limit) * 100 if limit > 0 else 0
        
        action = "pass"
        if pct > 100:
            action = "block"
        elif pct > 85:
            action = "compress"
        elif pct > 70:
            action = "warn"
            
        return {
            "action": action,
            "tokens": tokens,
            "limit": limit,
            "pct_used": round(pct, 1),
            "label": label
        }

    def trim_text(
        self,
        text: str,
        model: str | None = None,
        max_tokens: int | None = None,
        keep: str = "tail",
        budget_fraction: float = 0.7,
    ) -> str:
        """
        Trim text to fit within a fraction of the model context or a fixed max_tokens.
        """
        m = model or self.model
        limit = max_tokens or int(_usable_limit(m) * budget_fraction)
        target_chars = int(limit * _CHARS_PER_TOKEN)

        if len(text) <= target_chars:
            return text

        notice = f"\n[... {len(text) - target_chars} chars truncated for {m} budget ...]\n"

        if keep == "head":
            return text[:target_chars] + notice
        elif keep == "tail":
            return notice + text[-target_chars:]
        return notice + text[-target_chars:]

    def compress_tool_output(self, output: str, max_lines: int = 20) -> str:
        """
        Compress verbose tool output (Rule 1 — Tool Output Compression).
        Keeps first 5 and last 15 lines, replaces middle with a count.
        """
        lines = output.strip().splitlines()
        if len(lines) <= max_lines:
            return output

        head = lines[:5]
        tail = lines[-(max_lines - 5):]
        omitted = len(lines) - max_lines
        return "\n".join(head) + f"\n[... {omitted} lines omitted ...]\n" + "\n".join(tail)

    def estimate(self, text: str) -> int:
        """Estimate token count for text."""
        return _estimate_tokens(text)

    def set_model(self, model: str) -> None:
        """Update the model and recalculate limit."""
        self.model = model
        self.limit = _usable_limit(model)


def create(model: str = "default") -> ContextWindowManager:
    """Factory — called by executor's _load_cw() helper."""
    return ContextWindowManager(model)
