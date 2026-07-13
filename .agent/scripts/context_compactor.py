"""
context_compactor.py — Offline Context Compaction (Post-Task)

Uses tiktoken for accurate token counting + heuristic-based local compaction.
No API calls — fully offline. References Claude documentation:
  - Token Counting: platform.claude.com/docs/build-with-claude/token-counting
  - Context Editing: platform.claude.com/docs/build-with-claude/context-editing
  - Compaction: platform.claude.com/docs/build-with-claude/compaction

Strategies (free, offline):
  1. Tool result clearing — discard processed outputs, keep structured state
  2. Message pruning — keep N most recent turns, drop old ones
  3. Heuristic summarization — lightweight summary of discarded turns

USAGE:
  compactor = ContextCompactor()
  stats = compactor.estimate_tokens(conversation_history)
  if stats['tokens_used'] > 0.85 * stats['limit']:
      compacted = compactor.compact(conversation_history, strategy='tool_clearing')
      should_rotate = compactor.check_rotation_threshold(compacted)
"""
import logging
import json
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import re
log = logging.getLogger(__name__)
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    log.warning('tiktoken not found; using heuristic token estimation (accuracy ~80%)')

@dataclass
class TokenStats:
    """Token usage and threshold info."""
    tokens_used: int
    tokens_limit: int
    pct_used: float
    tokens_remaining: int
    can_rotate: bool

class ContextCompactor:
    """Offline token counting + heuristic compaction without API calls."""

    def __init__(self, model: str='claude-3-5-sonnet', token_limit: int=200000):
        """
        Args:
            model: Claude model name (for token estimation)
            token_limit: Max context window (default: Claude 3.5 Sonnet = 200k)
        """
        self.model = model
        self.token_limit = token_limit
        self.encoder = None
        if HAS_TIKTOKEN:
            try:
                self.encoder = tiktoken.get_encoding('cl100k_base')
                log.info(f'[compactor] tiktoken encoder loaded for {model}')
            except Exception as e:
                log.warning(f'[compactor] tiktoken init failed: {e}; using heuristic')
                self.encoder = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text. Returns accurate count if tiktoken available."""
        if not text:
            return 0
        if self.encoder:
            try:
                return len(self.encoder.encode(text))
            except Exception as e:
                log.debug(f'[compactor] tiktoken count failed: {e}')
        return max(1, len(text) // 3)

    def estimate_tokens(self, conversation: list[dict]) -> TokenStats:
        """
        Estimate total tokens in conversation history.

        Args:
            conversation: List of message dicts with 'role', 'content', 'tool_use_id', etc.

        Returns:
            TokenStats with usage summary
        """
        total = 0
        for msg in conversation:
            role_tokens = self.count_tokens(f'role:{msg.get('role', 'user')}')
            content = msg.get('content', '')
            if isinstance(content, str):
                content_tokens = self.count_tokens(content)
            elif isinstance(content, list):
                content_tokens = sum((self.count_tokens(json.dumps(block)) if isinstance(block, dict) else self.count_tokens(str(block)) for block in content))
            else:
                content_tokens = self.count_tokens(json.dumps(content))
            total += role_tokens + content_tokens + 5
        pct_used = total / self.token_limit * 100 if self.token_limit > 0 else 0
        can_rotate = pct_used > 85
        return TokenStats(tokens_used=total, tokens_limit=self.token_limit, pct_used=pct_used, tokens_remaining=self.token_limit - total, can_rotate=can_rotate)

    def compact(self, conversation: list[dict], strategy: str='tool_clearing', keep_last_n_turns: int=10) -> list[dict]:
        """
        Compact conversation using specified strategy (offline, no API calls).

        Args:
            conversation: Message history
            strategy: 'tool_clearing' | 'message_pruning' | 'combined'
            keep_last_n_turns: For message_pruning, how many recent turns to keep

        Returns:
            Compacted conversation (original untouched)
        """
        if not conversation:
            return []
        compacted = [msg.copy() for msg in conversation]
        if strategy in {'tool_clearing', 'combined'}:
            compacted = self._clear_tool_results(compacted)
            # After removing verbose tool outputs, further trim to keep only
            # assistant thinking/exploration blocks and explicit user intent/requests.
            compacted = self._retain_thinking_and_intent(compacted)
        if strategy in {'message_pruning', 'combined'}:
            compacted = self._prune_old_turns(compacted, keep_last_n=keep_last_n_turns)
        tokens_before = self.estimate_tokens(conversation).tokens_used
        tokens_after = self.estimate_tokens(compacted).tokens_used
        reduction_pct = (tokens_before - tokens_after) / max(1, tokens_before) * 100
        log.info(f'[compactor] {strategy}: {tokens_before:,} → {tokens_after:,} tokens ({reduction_pct:.1f}% reduction)')
        return compacted

    def _clear_tool_results(self, conversation: list[dict]) -> list[dict]:
        """Remove tool result content blocks, keeping only metadata."""
        compacted = []
        # Opt-in raw deletion via env var
        safe_delete_raw = os.environ.get('AB_SAFE_DELETE_RAW', 'false').lower() in ('1', 'true', 'yes')

        # Try to use context_intel summarizer + scratchpad if available
        have_context_intel = False
        summarize_tool_output = None
        store_tool_summary = None
        Scratchpad = None
        ExecutionLearner = None
        NegativeMemory = None
        try:
            # runtime package import (module lives under .agent/runtime/context_intel)
            from context_intel import summarize_tool_output as _summ, store_tool_summary as _store
            from context_intel import Scratchpad as _sp
            # optional learning/penalty helpers
            try:
                from context_intel import ExecutionLearner as _el, NegativeMemory as _nm
                ExecutionLearner = _el
                NegativeMemory = _nm
            except Exception:
                ExecutionLearner = None
                NegativeMemory = None
            summarize_tool_output = _summ
            store_tool_summary = _store
            Scratchpad = _sp
            have_context_intel = True
        except Exception:
            have_context_intel = False

        sp = None
        el = None
        nm = None
        if have_context_intel:
            try:
                sp = Scratchpad()
            except Exception:
                sp = None
            try:
                if ExecutionLearner:
                    el = ExecutionLearner()
            except Exception:
                el = None
            try:
                if NegativeMemory:
                    nm = NegativeMemory()
            except Exception:
                nm = None

        for msg in conversation:
            msg_copy = msg.copy()
            content = msg_copy.get('content', [])
            if isinstance(content, list):
                filtered = []
                for block in content:
                    # Dict blocks carry structured info about tool runs, diffs, or annotations
                    if isinstance(block, dict):
                        block_type = (block.get('type') or '').lower()
                        is_tool_block = False
                        if block_type == 'tool_result' or any(k in block for k in ('cmd', 'command', 'stdout', 'stderr', 'exit_code', 'returncode')) or block_type in {'cli', 'command', 'shell', 'process', 'subprocess'}:
                            is_tool_block = True

                        if is_tool_block:
                            # try to extract raw text from common fields
                            raw_text = ''
                            if isinstance(block.get('content'), str) and block.get('content').strip():
                                raw_text = block.get('content')
                            else:
                                parts = []
                                if block.get('stdout'):
                                    parts.append(str(block.get('stdout')))
                                if block.get('stderr'):
                                    parts.append(str(block.get('stderr')))
                                if block.get('text') and isinstance(block.get('text'), str):
                                    parts.append(block.get('text'))
                                raw_text = '\n'.join([p for p in parts if p])

                            if have_context_intel and raw_text:
                                try:
                                    summary = summarize_tool_output(raw_text)
                                    # store summary (jsonl) and optionally delete raw file if provided and safe
                                    store_tool_summary(summary, block.get('tool') or block.get('name') or 'unknown', delete_raw_path=block.get('raw_path') or block.get('output_file'), safe_delete=safe_delete_raw)
                                    filtered.append({
                                        'type': 'tool_result_summary',
                                        'tool': block.get('tool') or block.get('name'),
                                        'tool_use_id': block.get('tool_use_id'),
                                        'is_error': summary.get('outcome') != 'success',
                                        'summary': summary,
                                    })
                                    # record to scratchpad
                                    if sp:
                                        try:
                                            sp.add_decision(tool=block.get('tool') or block.get('name') or 'unknown', action='ran', reason=summary.get('failure_reason') if summary.get('outcome') != 'success' else 'completed', skipped=False)
                                            sp.add_outcome(success=(summary.get('outcome') == 'success'), note=summary.get('failure_reason') if summary.get('outcome') != 'success' else 'ok')
                                        except Exception:
                                            pass
                                    # record to execution learner (simple learning loop)
                                    if el:
                                        try:
                                            el.record_tool_run(block.get('tool') or block.get('name') or 'unknown', success=(summary.get('outcome') == 'success'), score=summary.get('confidence'), note=summary.get('failure_reason'))
                                        except Exception:
                                            pass
                                    # update negative memory for failures
                                    if nm and summary.get('outcome') != 'success':
                                        try:
                                            nm.penalize_tool(block.get('tool') or block.get('name') or 'unknown', penalty=1.0, reason=summary.get('failure_reason'))
                                        except Exception:
                                            pass
                                except Exception:
                                    # fallback to minimal metadata
                                    filtered.append({
                                        'type': 'tool_result',
                                        'tool': block.get('tool') or block.get('name'),
                                        'tool_use_id': block.get('tool_use_id'),
                                        'is_error': block.get('is_error', False),
                                        'content': None,
                                    })
                            else:
                                # fallback minimal metadata
                                filtered.append({
                                    'type': 'tool_result',
                                    'tool': block.get('tool') or block.get('name'),
                                    'tool_use_id': block.get('tool_use_id'),
                                    'is_error': block.get('is_error', False),
                                    'content': None,
                                })
                        else:
                            filtered.append(block)
                    else:
                        filtered.append(block)
                msg_copy['content'] = filtered
            compacted.append(msg_copy)

        # persist scratchpad if used
        if sp:
            try:
                sp.save()
            except Exception:
                pass

        # compute and persist tool weights via execution learner (best-effort)
        if el:
            try:
                el.compute_and_persist_weights(half_life_days=30.0, negative_memory=nm)
            except Exception:
                pass

        return compacted

    def _is_diff_block(self, block: dict) -> bool:
        """Heuristic: detect unified diffs / patches in block content."""
        if not isinstance(block, dict):
            return False
        t = (block.get('type') or '').lower()
        if t in {'file_diff', 'diff', 'patch'}:
            return True
        content = block.get('content') or block.get('text') or ''
        if isinstance(content, str) and ("diff --git" in content or content.strip().startswith('*** Begin Patch') or content.strip().startswith('+++') or content.strip().startswith('---')):
            return True
        return False

    def _summarize_diff_text(self, text: str, max_lines: int = 8) -> dict:
        """Return a compact summary of a unified diff: files changed, +/-, snippet."""
        if not text:
            return {'files': [], 'summary': 'empty', 'snippet': ''}
        lines = text.splitlines()
        files = []
        adds = 0
        dels = 0
        snippet_lines = []
        for ln in lines:
            if ln.startswith('diff --git'):
                parts = ln.split()
                if len(parts) >= 3:
                    files.append(parts[-1])
            if ln.startswith('+') and not ln.startswith('+++'):
                adds += 1
            if ln.startswith('-') and not ln.startswith('---'):
                dels += 1
            if len(snippet_lines) < max_lines:
                snippet_lines.append(ln)
        summary = f"files={len(files)} +{adds} -{dels}"
        snippet = '\n'.join(snippet_lines)
        return {'files': files, 'summary': summary, 'snippet': snippet}

    def _retain_thinking_and_intent(self, conversation: list[dict], keep_diff_snippet_lines: int = 8) -> list[dict]:
        """Retain only system messages, assistant thinking/exploration blocks and explicit user intent/requests.

        Tool outputs, CLI runs and large `tool_result` blocks are removed (metadata preserved).
        File diffs are summarized and kept as compact metadata (not full code blobs).
        """
        kept = []
        keyword_re = re.compile(r"\b(thoughts?|thinking|plan|intent|request|direction|explor|analysis|goal|ask|task)\b", re.I)

        for msg in conversation:
            # Always keep system messages
            if msg.get('role') == 'system':
                kept.append(msg)
                continue

            msg_copy = msg.copy()
            content = msg_copy.get('content', [])
            # Normalize string content into list for unified handling
            blocks = content if isinstance(content, list) else [content]
            filtered_blocks = []

            for block in blocks:
                # Structured blocks
                if isinstance(block, dict):
                    # Keep thinking/exploration/reasoning blocks verbatim
                    btype = (block.get('type') or '').lower()
                    if btype in {'thinking', 'thought', 'analysis', 'reasoning', 'exploration', 'explore', 'plan'}:
                        filtered_blocks.append(block)
                        continue

                    # Keep explicit intent/request/direction blocks
                    if btype in {'intent', 'request', 'user_intent', 'user_request', 'direction', 'instruction', 'goal'}:
                        filtered_blocks.append(block)
                        continue

                    # Keep file diffs as summaries
                    if self._is_diff_block(block):
                        text = block.get('content') or block.get('text') or ''
                        summary = self._summarize_diff_text(text, max_lines=keep_diff_snippet_lines)
                        filtered_blocks.append({'type': 'file_diff_summary', 'files': summary['files'], 'summary': summary['summary'], 'snippet': summary['snippet']})
                        continue

                    # Tool / CLI outputs are dropped to metadata-only in earlier pass
                    # But if they contain explicit intent-like keywords, keep a trimmed note
                    text = ''
                    if isinstance(block.get('content'), str):
                        text = block.get('content')
                    elif isinstance(block.get('text'), str):
                        text = block.get('text')
                    if text and keyword_re.search(text):
                        # keep a short excerpt
                        excerpt = '\n'.join(text.splitlines()[:3])
                        filtered_blocks.append({'type': 'excerpt', 'content': excerpt})
                        continue

                    # Otherwise skip the block (tool/cli/verbose output)
                    continue

                # Plain text blocks: keep if they include intent/thinking keywords or diff markers
                if isinstance(block, str):
                    if keyword_re.search(block) or 'diff --git' in block or block.strip().startswith('*** Begin Patch'):
                        # trim long text
                        trimmed = '\n'.join(block.splitlines()[:6])
                        filtered_blocks.append(trimmed)
                        continue
                    # otherwise drop
                    continue

            if filtered_blocks:
                msg_copy['content'] = filtered_blocks
                kept.append(msg_copy)

        return kept

    def _prune_old_turns(self, conversation: list[dict], keep_last_n: int=10) -> list[dict]:
        """Keep system message + last N turns; discard older turns."""
        if len(conversation) <= keep_last_n:
            return conversation
        compacted = []
        for msg in conversation:
            if msg.get('role') == 'system':
                compacted.append(msg)
        turns = [msg for msg in conversation if msg.get('role') != 'system']
        compacted.extend(turns[-keep_last_n:])
        return compacted

    def check_rotation_threshold(self, conversation: list[dict], threshold_pct: float=85.0) -> bool:
        """
        Check if conversation exceeds rotation threshold.

        Returns:
            True if tokens_used / tokens_limit >= threshold_pct
        """
        stats = self.estimate_tokens(conversation)
        return stats.pct_used >= threshold_pct

@dataclass
class TokenStats__from_archived_9085:
    """Token usage and threshold info."""
    tokens_used: int
    tokens_limit: int
    pct_used: float
    tokens_remaining: int
    can_rotate: bool

class ContextCompactor__from_archived_9085:
    """Offline token counting + heuristic compaction without API calls."""

    def __init__(self, model: str='claude-3-5-sonnet', token_limit: int=200000):
        """
        Args:
            model: Claude model name (for token estimation)
            token_limit: Max context window (default: Claude 3.5 Sonnet = 200k)
        """
        self.model = model
        self.token_limit = token_limit
        self.encoder = None
        if HAS_TIKTOKEN:
            try:
                self.encoder = tiktoken.get_encoding('cl100k_base')
                log.info(f'[compactor] tiktoken encoder loaded for {model}')
            except Exception as e:
                log.warning(f'[compactor] tiktoken init failed: {e}; using heuristic')
                self.encoder = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text. Returns accurate count if tiktoken available."""
        if not text:
            return 0
        if self.encoder:
            try:
                return len(self.encoder.encode(text))
            except Exception as e:
                log.debug(f'[compactor] tiktoken count failed: {e}')
        return max(1, len(text) // 3)

    def estimate_tokens(self, conversation: list[dict]) -> TokenStats:
        """
        Estimate total tokens in conversation history.

        Args:
            conversation: List of message dicts with 'role', 'content', 'tool_use_id', etc.

        Returns:
            TokenStats with usage summary
        """
        total = 0
        for msg in conversation:
            role_tokens = self.count_tokens(f'role:{msg.get('role', 'user')}')
            content = msg.get('content', '')
            if isinstance(content, str):
                content_tokens = self.count_tokens(content)
            elif isinstance(content, list):
                content_tokens = sum((self.count_tokens(json.dumps(block)) if isinstance(block, dict) else self.count_tokens(str(block)) for block in content))
            else:
                content_tokens = self.count_tokens(json.dumps(content))
            total += role_tokens + content_tokens + 5
        pct_used = total / self.token_limit * 100 if self.token_limit > 0 else 0
        can_rotate = pct_used > 85
        return TokenStats(tokens_used=total, tokens_limit=self.token_limit, pct_used=pct_used, tokens_remaining=self.token_limit - total, can_rotate=can_rotate)

    def compact(self, conversation: list[dict], strategy: str='tool_clearing', keep_last_n_turns: int=10) -> list[dict]:
        """
        Compact conversation using specified strategy (offline, no API calls).

        Args:
            conversation: Message history
            strategy: 'tool_clearing' | 'message_pruning' | 'combined'
            keep_last_n_turns: For message_pruning, how many recent turns to keep

        Returns:
            Compacted conversation (original untouched)
        """
        if not conversation:
            return []
        compacted = [msg.copy() for msg in conversation]
        if strategy in {'tool_clearing', 'combined'}:
            compacted = self._clear_tool_results(compacted)
        if strategy in {'message_pruning', 'combined'}:
            compacted = self._prune_old_turns(compacted, keep_last_n=keep_last_n_turns)
        tokens_before = self.estimate_tokens(conversation).tokens_used
        tokens_after = self.estimate_tokens(compacted).tokens_used
        reduction_pct = (tokens_before - tokens_after) / max(1, tokens_before) * 100
        log.info(f'[compactor] {strategy}: {tokens_before:,} → {tokens_after:,} tokens ({reduction_pct:.1f}% reduction)')
        return compacted

    def _clear_tool_results(self, conversation: list[dict]) -> list[dict]:
        """Remove tool result content blocks, keeping only metadata."""
        compacted = []
        for msg in conversation:
            msg_copy = msg.copy()
            content = msg_copy.get('content', [])
            if isinstance(content, list):
                filtered = []
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get('type')
                        if block_type == 'tool_result':
                            filtered.append({'type': 'tool_result', 'tool_use_id': block.get('tool_use_id'), 'is_error': block.get('is_error', False), 'content': '[tool output cleared for compaction]' if block.get('content') else None})
                        else:
                            filtered.append(block)
                    else:
                        filtered.append(block)
                msg_copy['content'] = filtered
            compacted.append(msg_copy)
        return compacted

    def _prune_old_turns(self, conversation: list[dict], keep_last_n: int=10) -> list[dict]:
        """Keep system message + last N turns; discard older turns."""
        if len(conversation) <= keep_last_n:
            return conversation
        compacted = []
        for msg in conversation:
            if msg.get('role') == 'system':
                compacted.append(msg)
        turns = [msg for msg in conversation if msg.get('role') != 'system']
        compacted.extend(turns[-keep_last_n:])
        return compacted

    def check_rotation_threshold(self, conversation: list[dict], threshold_pct: float=85.0) -> bool:
        """
        Check if conversation exceeds rotation threshold.

        Returns:
            True if tokens_used / tokens_limit >= threshold_pct
        """
        stats = self.estimate_tokens(conversation)
        return stats.pct_used >= threshold_pct

@dataclass
class TokenStats__from_archived_4598:
    """Token usage and threshold info."""
    tokens_used: int
    tokens_limit: int
    pct_used: float
    tokens_remaining: int
    can_rotate: bool

class ContextCompactor__from_archived_4598:
    """Offline token counting + heuristic compaction without API calls."""

    def __init__(self, model: str='claude-3-5-sonnet', token_limit: int=200000):
        """
        Args:
            model: Claude model name (for token estimation)
            token_limit: Max context window (default: Claude 3.5 Sonnet = 200k)
        """
        self.model = model
        self.token_limit = token_limit
        self.encoder = None
        if HAS_TIKTOKEN:
            try:
                self.encoder = tiktoken.get_encoding('cl100k_base')
                log.info(f'[compactor] tiktoken encoder loaded for {model}')
            except Exception as e:
                log.warning(f'[compactor] tiktoken init failed: {e}; using heuristic')
                self.encoder = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text. Returns accurate count if tiktoken available."""
        if not text:
            return 0
        if self.encoder:
            try:
                return len(self.encoder.encode(text))
            except Exception as e:
                log.debug(f'[compactor] tiktoken count failed: {e}')
        return max(1, len(text) // 3)

    def estimate_tokens(self, conversation: list[dict]) -> TokenStats:
        """
        Estimate total tokens in conversation history.

        Args:
            conversation: List of message dicts with 'role', 'content', 'tool_use_id', etc.

        Returns:
            TokenStats with usage summary
        """
        total = 0
        for msg in conversation:
            role_tokens = self.count_tokens(f'role:{msg.get('role', 'user')}')
            content = msg.get('content', '')
            if isinstance(content, str):
                content_tokens = self.count_tokens(content)
            elif isinstance(content, list):
                content_tokens = sum((self.count_tokens(json.dumps(block)) if isinstance(block, dict) else self.count_tokens(str(block)) for block in content))
            else:
                content_tokens = self.count_tokens(json.dumps(content))
            total += role_tokens + content_tokens + 5
        pct_used = total / self.token_limit * 100 if self.token_limit > 0 else 0
        can_rotate = pct_used > 85
        return TokenStats(tokens_used=total, tokens_limit=self.token_limit, pct_used=pct_used, tokens_remaining=self.token_limit - total, can_rotate=can_rotate)

    def compact(self, conversation: list[dict], strategy: str='tool_clearing', keep_last_n_turns: int=10) -> list[dict]:
        """
        Compact conversation using specified strategy (offline, no API calls).

        Args:
            conversation: Message history
            strategy: 'tool_clearing' | 'message_pruning' | 'combined'
            keep_last_n_turns: For message_pruning, how many recent turns to keep

        Returns:
            Compacted conversation (original untouched)
        """
        if not conversation:
            return []
        compacted = [msg.copy() for msg in conversation]
        if strategy in {'tool_clearing', 'combined'}:
            compacted = self._clear_tool_results(compacted)
        if strategy in {'message_pruning', 'combined'}:
            compacted = self._prune_old_turns(compacted, keep_last_n=keep_last_n_turns)
        tokens_before = self.estimate_tokens(conversation).tokens_used
        tokens_after = self.estimate_tokens(compacted).tokens_used
        reduction_pct = (tokens_before - tokens_after) / max(1, tokens_before) * 100
        log.info(f'[compactor] {strategy}: {tokens_before:,} → {tokens_after:,} tokens ({reduction_pct:.1f}% reduction)')
        return compacted

    def _clear_tool_results(self, conversation: list[dict]) -> list[dict]:
        """Remove tool result content blocks, keeping only metadata."""
        compacted = []
        for msg in conversation:
            msg_copy = msg.copy()
            content = msg_copy.get('content', [])
            if isinstance(content, list):
                filtered = []
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get('type')
                        if block_type == 'tool_result':
                            filtered.append({'type': 'tool_result', 'tool_use_id': block.get('tool_use_id'), 'is_error': block.get('is_error', False), 'content': '[tool output cleared for compaction]' if block.get('content') else None})
                        else:
                            filtered.append(block)
                    else:
                        filtered.append(block)
                msg_copy['content'] = filtered
            compacted.append(msg_copy)
        return compacted

    def _prune_old_turns(self, conversation: list[dict], keep_last_n: int=10) -> list[dict]:
        """Keep system message + last N turns; discard older turns."""
        if len(conversation) <= keep_last_n:
            return conversation
        compacted = []
        for msg in conversation:
            if msg.get('role') == 'system':
                compacted.append(msg)
        turns = [msg for msg in conversation if msg.get('role') != 'system']
        compacted.extend(turns[-keep_last_n:])
        return compacted

    def check_rotation_threshold(self, conversation: list[dict], threshold_pct: float=85.0) -> bool:
        """
        Check if conversation exceeds rotation threshold.

        Returns:
            True if tokens_used / tokens_limit >= threshold_pct
        """
        stats = self.estimate_tokens(conversation)
        return stats.pct_used >= threshold_pct

# === MERGED FROM C:\Users\AmitMohanty\.vscode\brain\.agent\tmp\moved_duplicates\20260328T133840Z\context_compactor.py ===
# """
# context_compactor.py — Offline Context Compaction (Post-Task)
# 
# Uses tiktoken for accurate token counting + heuristic-based local compaction.
# No API calls — fully offline. References Claude documentation:
#   - Token Counting: platform.claude.com/docs/build-with-claude/token-counting
#   - Context Editing: platform.claude.com/docs/build-with-claude/context-editing
#   - Compaction: platform.claude.com/docs/build-with-claude/compaction
# 
# Strategies (free, offline):
#   1. Tool result clearing — discard processed outputs, keep structured state
#   2. Message pruning — keep N most recent turns, drop old ones
#   3. Heuristic summarization — lightweight summary of discarded turns
# 
# USAGE:
#   compactor = ContextCompactor()
#   stats = compactor.estimate_tokens(conversation_history)
#   if stats['tokens_used'] > 0.85 * stats['limit']:
#       compacted = compactor.compact(conversation_history, strategy='tool_clearing')
#       should_rotate = compactor.check_rotation_threshold(compacted)
# """
# 
# import logging
# import json
# from pathlib import Path
# from typing import Optional
# from dataclasses import dataclass
# 
# log = logging.getLogger(__name__)
# 
# # Attempt to import tiktoken; fallback to heuristic estimation if unavailable
# try:
#     import tiktoken
#     HAS_TIKTOKEN = True
# except ImportError:
#     HAS_TIKTOKEN = False
#     log.warning("tiktoken not found; using heuristic token estimation (accuracy ~80%)")
# 
# 
# @dataclass
# class TokenStats:
#     """Token usage and threshold info."""
#     tokens_used: int
#     tokens_limit: int
#     pct_used: float
#     tokens_remaining: int
#     can_rotate: bool
# 
# 
# class ContextCompactor:
#     """Offline token counting + heuristic compaction without API calls."""
# 
#     def __init__(self, model: str = "claude-3-5-sonnet", token_limit: int = 200_000):
#         """
#         Args:
#             model: Claude model name (for token estimation)
#             token_limit: Max context window (default: Claude 3.5 Sonnet = 200k)
#         """
#         self.model = model
#         self.token_limit = token_limit
#         self.encoder = None
# 
#         if HAS_TIKTOKEN:
#             try:
#                 # Use cl100k_base (accurate for Claude) instead of gpt2
#                 self.encoder = tiktoken.get_encoding("cl100k_base")
#                 log.info(f"[compactor] tiktoken encoder loaded for {model}")
#             except Exception as e:
#                 log.warning(f"[compactor] tiktoken init failed: {e}; using heuristic")
#                 self.encoder = None
# 
#     def count_tokens(self, text: str) -> int:
#         """Count tokens in text. Returns accurate count if tiktoken available."""
#         if not text:
#             return 0
# 
#         if self.encoder:
#             try:
#                 return len(self.encoder.encode(text))
#             except Exception as e:
#                 log.debug(f"[compactor] tiktoken count failed: {e}")
# 
#         # Fallback: heuristic (~1 token ≈ 4 chars, but more accurate for prose)
#         # For JSON/code, typically 1 token ≈ 3 chars
#         return max(1, len(text) // 3)
# 
#     def estimate_tokens(self, conversation: list[dict]) -> TokenStats:
#         """
#         Estimate total tokens in conversation history.
# 
#         Args:
#             conversation: List of message dicts with 'role', 'content', 'tool_use_id', etc.
# 
#         Returns:
#             TokenStats with usage summary
#         """
#         total = 0
#         for msg in conversation:
#             role_tokens = self.count_tokens(f"role:{msg.get('role', 'user')}")
#             content = msg.get("content", "")
# 
#             if isinstance(content, str):
#                 content_tokens = self.count_tokens(content)
#             elif isinstance(content, list):
#                 # List of content blocks (text + tool_use)
#                 content_tokens = sum(
#                     self.count_tokens(json.dumps(block)) if isinstance(block, dict)
#                     else self.count_tokens(str(block))
#                     for block in content
#                 )
#             else:
#                 content_tokens = self.count_tokens(json.dumps(content))
# 
#             total += role_tokens + content_tokens + 5  # +5 for msg overhead
# 
#         pct_used = (total / self.token_limit) * 100 if self.token_limit > 0 else 0
#         can_rotate = pct_used > 85  # Rotate at 85% capacity
# 
#         return TokenStats(
#             tokens_used=total,
#             tokens_limit=self.token_limit,
#             pct_used=pct_used,
#             tokens_remaining=self.token_limit - total,
#             can_rotate=can_rotate,
#         )
# 
#     def compact(
#         self,
#         conversation: list[dict],
#         strategy: str = "tool_clearing",
#         keep_last_n_turns: int = 10,
#     ) -> list[dict]:
#         """
#         Compact conversation using specified strategy (offline, no API calls).
# 
#         Args:
#             conversation: Message history
#             strategy: 'tool_clearing' | 'message_pruning' | 'combined'
#             keep_last_n_turns: For message_pruning, how many recent turns to keep
# 
#         Returns:
#             Compacted conversation (original untouched)
#         """
#         if not conversation:
#             return []
# 
#         compacted = [msg.copy() for msg in conversation]
# 
#         if strategy in {"tool_clearing", "combined"}:
#             compacted = self._clear_tool_results(compacted)
# 
#         if strategy in {"message_pruning", "combined"}:
#             compacted = self._prune_old_turns(compacted, keep_last_n=keep_last_n_turns)
# 
#         tokens_before = self.estimate_tokens(conversation).tokens_used
#         tokens_after = self.estimate_tokens(compacted).tokens_used
#         reduction_pct = ((tokens_before - tokens_after) / max(1, tokens_before)) * 100
# 
#         log.info(
#             f"[compactor] {strategy}: {tokens_before:,} → {tokens_after:,} tokens "
#             f"({reduction_pct:.1f}% reduction)"
#         )
#         return compacted
# 
#     def _clear_tool_results(self, conversation: list[dict]) -> list[dict]:
#         """Remove tool result content blocks, keeping only metadata."""
#         compacted = []
#         for msg in conversation:
#             msg_copy = msg.copy()
#             content = msg_copy.get("content", [])
# 
#             if isinstance(content, list):
#                 # Filter: keep text blocks and tool_use, discard tool_result body text
#                 filtered = []
#                 for block in content:
#                     if isinstance(block, dict):
#                         block_type = block.get("type")
#                         if block_type == "tool_result":
#                             # Keep only tool_use_id and is_error, discard content
#                             filtered.append({
#                                 "type": "tool_result",
#                                 "tool_use_id": block.get("tool_use_id"),
#                                 "is_error": block.get("is_error", False),
#                                 "content": "[tool output cleared for compaction]" if block.get("content") else None
#                             })
#                         else:
#                             filtered.append(block)
#                     else:
#                         filtered.append(block)
#                 msg_copy["content"] = filtered
#             compacted.append(msg_copy)
# 
#         return compacted
# 
#     def _prune_old_turns(self, conversation: list[dict], keep_last_n: int = 10) -> list[dict]:
#         """Keep system message + last N turns; discard older turns."""
#         if len(conversation) <= keep_last_n:
#             return conversation
# 
#         compacted = []
# 
#         # Preserve system messages (always keep)
#         for msg in conversation:
#             if msg.get("role") == "system":
#                 compacted.append(msg)
# 
#         # Keep last N user/assistant turns
#         turns = [msg for msg in conversation if msg.get("role") != "system"]
#         compacted.extend(turns[-keep_last_n:])
# 
#         return compacted
# 
#     def check_rotation_threshold(self, conversation: list[dict], threshold_pct: float = 85.0) -> bool:
#         """
#         Check if conversation exceeds rotation threshold.
# 
#         Returns:
#             True if tokens_used / tokens_limit >= threshold_pct
#         """
#         stats = self.estimate_tokens(conversation)
#         return stats.pct_used >= threshold_pct


# === MERGED FROM C:\Users\AmitMohanty\.vscode\brain\.agent\tmp\moved_duplicates\20260328T133840Z\context_compactor.py.1 ===
# """
# context_compactor.py — Offline Context Compaction (Post-Task)
# 
# Uses tiktoken for accurate token counting + heuristic-based local compaction.
# No API calls — fully offline. References Claude documentation:
#   - Token Counting: platform.claude.com/docs/build-with-claude/token-counting
#   - Context Editing: platform.claude.com/docs/build-with-claude/context-editing
#   - Compaction: platform.claude.com/docs/build-with-claude/compaction
# 
# Strategies (free, offline):
#   1. Tool result clearing — discard processed outputs, keep structured state
#   2. Message pruning — keep N most recent turns, drop old ones
#   3. Heuristic summarization — lightweight summary of discarded turns
# 
# USAGE:
#   compactor = ContextCompactor()
#   stats = compactor.estimate_tokens(conversation_history)
#   if stats['tokens_used'] > 0.85 * stats['limit']:
#       compacted = compactor.compact(conversation_history, strategy='tool_clearing')
#       should_rotate = compactor.check_rotation_threshold(compacted)
# """
# 
# import logging
# import json
# from pathlib import Path
# from typing import Optional
# from dataclasses import dataclass
# 
# log = logging.getLogger(__name__)
# 
# # Attempt to import tiktoken; fallback to heuristic estimation if unavailable
# try:
#     import tiktoken
#     HAS_TIKTOKEN = True
# except ImportError:
#     HAS_TIKTOKEN = False
#     log.warning("tiktoken not found; using heuristic token estimation (accuracy ~80%)")
# 
# 
# @dataclass
# class TokenStats:
#     """Token usage and threshold info."""
#     tokens_used: int
#     tokens_limit: int
#     pct_used: float
#     tokens_remaining: int
#     can_rotate: bool
# 
# 
# class ContextCompactor:
#     """Offline token counting + heuristic compaction without API calls."""
# 
#     def __init__(self, model: str = "claude-3-5-sonnet", token_limit: int = 200_000):
#         """
#         Args:
#             model: Claude model name (for token estimation)
#             token_limit: Max context window (default: Claude 3.5 Sonnet = 200k)
#         """
#         self.model = model
#         self.token_limit = token_limit
#         self.encoder = None
# 
#         if HAS_TIKTOKEN:
#             try:
#                 # Use cl100k_base (accurate for Claude) instead of gpt2
#                 self.encoder = tiktoken.get_encoding("cl100k_base")
#                 log.info(f"[compactor] tiktoken encoder loaded for {model}")
#             except Exception as e:
#                 log.warning(f"[compactor] tiktoken init failed: {e}; using heuristic")
#                 self.encoder = None
# 
#     def count_tokens(self, text: str) -> int:
#         """Count tokens in text. Returns accurate count if tiktoken available."""
#         if not text:
#             return 0
# 
#         if self.encoder:
#             try:
#                 return len(self.encoder.encode(text))
#             except Exception as e:
#                 log.debug(f"[compactor] tiktoken count failed: {e}")
# 
#         # Fallback: heuristic (~1 token ≈ 4 chars, but more accurate for prose)
#         # For JSON/code, typically 1 token ≈ 3 chars
#         return max(1, len(text) // 3)
# 
#     def estimate_tokens(self, conversation: list[dict]) -> TokenStats:
#         """
#         Estimate total tokens in conversation history.
# 
#         Args:
#             conversation: List of message dicts with 'role', 'content', 'tool_use_id', etc.
# 
#         Returns:
#             TokenStats with usage summary
#         """
#         total = 0
#         for msg in conversation:
#             role_tokens = self.count_tokens(f"role:{msg.get('role', 'user')}")
#             content = msg.get("content", "")
# 
#             if isinstance(content, str):
#                 content_tokens = self.count_tokens(content)
#             elif isinstance(content, list):
#                 # List of content blocks (text + tool_use)
#                 content_tokens = sum(
#                     self.count_tokens(json.dumps(block)) if isinstance(block, dict)
#                     else self.count_tokens(str(block))
#                     for block in content
#                 )
#             else:
#                 content_tokens = self.count_tokens(json.dumps(content))
# 
#             total += role_tokens + content_tokens + 5  # +5 for msg overhead
# 
#         pct_used = (total / self.token_limit) * 100 if self.token_limit > 0 else 0
#         can_rotate = pct_used > 85  # Rotate at 85% capacity
# 
#         return TokenStats(
#             tokens_used=total,
#             tokens_limit=self.token_limit,
#             pct_used=pct_used,
#             tokens_remaining=self.token_limit - total,
#             can_rotate=can_rotate,
#         )
# 
#     def compact(
#         self,
#         conversation: list[dict],
#         strategy: str = "tool_clearing",
#         keep_last_n_turns: int = 10,
#     ) -> list[dict]:
#         """
#         Compact conversation using specified strategy (offline, no API calls).
# 
#         Args:
#             conversation: Message history
#             strategy: 'tool_clearing' | 'message_pruning' | 'combined'
#             keep_last_n_turns: For message_pruning, how many recent turns to keep
# 
#         Returns:
#             Compacted conversation (original untouched)
#         """
#         if not conversation:
#             return []
# 
#         compacted = [msg.copy() for msg in conversation]
# 
#         if strategy in {"tool_clearing", "combined"}:
#             compacted = self._clear_tool_results(compacted)
# 
#         if strategy in {"message_pruning", "combined"}:
#             compacted = self._prune_old_turns(compacted, keep_last_n=keep_last_n_turns)
# 
#         tokens_before = self.estimate_tokens(conversation).tokens_used
#         tokens_after = self.estimate_tokens(compacted).tokens_used
#         reduction_pct = ((tokens_before - tokens_after) / max(1, tokens_before)) * 100
# 
#         log.info(
#             f"[compactor] {strategy}: {tokens_before:,} → {tokens_after:,} tokens "
#             f"({reduction_pct:.1f}% reduction)"
#         )
#         return compacted
# 
#     def _clear_tool_results(self, conversation: list[dict]) -> list[dict]:
#         """Remove tool result content blocks, keeping only metadata."""
#         compacted = []
#         for msg in conversation:
#             msg_copy = msg.copy()
#             content = msg_copy.get("content", [])
# 
#             if isinstance(content, list):
#                 # Filter: keep text blocks and tool_use, discard tool_result body text
#                 filtered = []
#                 for block in content:
#                     if isinstance(block, dict):
#                         block_type = block.get("type")
#                         if block_type == "tool_result":
#                             # Keep only tool_use_id and is_error, discard content
#                             filtered.append({
#                                 "type": "tool_result",
#                                 "tool_use_id": block.get("tool_use_id"),
#                                 "is_error": block.get("is_error", False),
#                                 "content": "[tool output cleared for compaction]" if block.get("content") else None
#                             })
#                         else:
#                             filtered.append(block)
#                     else:
#                         filtered.append(block)
#                 msg_copy["content"] = filtered
#             compacted.append(msg_copy)
# 
#         return compacted
# 
#     def _prune_old_turns(self, conversation: list[dict], keep_last_n: int = 10) -> list[dict]:
#         """Keep system message + last N turns; discard older turns."""
#         if len(conversation) <= keep_last_n:
#             return conversation
# 
#         compacted = []
# 
#         # Preserve system messages (always keep)
#         for msg in conversation:
#             if msg.get("role") == "system":
#                 compacted.append(msg)
# 
#         # Keep last N user/assistant turns
#         turns = [msg for msg in conversation if msg.get("role") != "system"]
#         compacted.extend(turns[-keep_last_n:])
# 
#         return compacted
# 
#     def check_rotation_threshold(self, conversation: list[dict], threshold_pct: float = 85.0) -> bool:
#         """
#         Check if conversation exceeds rotation threshold.
# 
#         Returns:
#             True if tokens_used / tokens_limit >= threshold_pct
#         """
#         stats = self.estimate_tokens(conversation)
#         return stats.pct_used >= threshold_pct
