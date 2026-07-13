"""
gpt5_adaptive_compactor.py — GPT-5-Mini Adaptive Thinking Context Compactor

Uses gpt-5-mini (free tier) with adaptive thinking for intelligent context compaction.
Implements Claude's context management architecture:
  - Thinking blocks automatically excluded from previous turns
  - Extended thinking tokens billed as output tokens only (fallback)
  - Effective context window = (input_tokens - previous_thinking_tokens) + current_turn_tokens
  - Token-efficient extensive reasoning without waste

References:
  - platform.claude.com/docs/build-with-claude/token-counting
  - platform.claude.com/docs/build-with-claude/context-windows
  - platform.claude.com/docs/build-with-claude/compaction
  - platform.claude.com/docs/build-with-claude/context-editing
  - platform.claude.com/docs/build-with-claude/prompt-caching

Strategy:
  1. Adaptive thinking (default): Fast, free, suitable for most compaction scenarios
  2. Extended thinking (fallback): Only when adaptive is insufficient or task is complex
  3. Thinking exclusion: Leverage automatic thinking block filtering in API

USAGE:
  compactor = GPT5AdaptiveCompactor()
  compacted = compactor.compact(
      conversation=history,
      token_budget=200_000,
      strategy='adaptive',  # or 'extended'
  )
  metadata = compactor.get_compaction_metadata()
"""
import logging
import json
import os
from pathlib import Path
from typing import Optional, TypedDict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
log = logging.getLogger(__name__)
BEAST_BACKEND = os.environ.get('BEAST_BACKEND', 'copilot-sdk')

class CompactionMetadata(TypedDict):
    """Metadata about compaction operation."""
    tokens_before: int
    tokens_after: int
    reduction_pct: float
    strategy_used: str
    thinking_used: bool
    thinking_tokens: int
    model: str
    timestamp: str
    success: bool

@dataclass
class CompactionResult:
    """Result of compaction operation."""
    conversation: list[dict]
    metadata: CompactionMetadata
    is_sufficient: bool

class GPT5AdaptiveCompactor:
    """
    Model-based context compactor using gpt-5-mini with adaptive thinking.
    
    Integrates with AutoBeast's model selection (via Copilot SDK / worker.py).
    Uses free gpt-5-mini tier; escalates to extended thinking only when needed.
    """

    def __init__(self, model: str='gpt-5-mini', token_limit: int=200000, compaction_threshold_pct: float=85.0, enable_extended_thinking: bool=True):
        """
        Args:
            model: Model to use (default: gpt-5-mini, free)
            token_limit: Maximum context window
            compaction_threshold_pct: Target threshold after compaction
            enable_extended_thinking: Allow fallback to extended thinking if adaptive insufficient
        """
        self.model = model
        self.token_limit = token_limit
        self.threshold_pct = compaction_threshold_pct
        self.enable_extended = enable_extended_thinking
        self.backend = BEAST_BACKEND
        log.info(f'[gpt5-compactor] initialized: model={model}, backend={self.backend}, token_limit={token_limit:,}, threshold={compaction_threshold_pct}%')

    def compact(self, conversation: list[dict], token_budget: Optional[int]=None, strategy: str='adaptive') -> CompactionResult:
        """
        Compact conversation using gpt-5-mini with adaptive thinking.

        Args:
            conversation: Message history
            token_budget: Optional override for token limit (uses self.token_limit if None)
            strategy: 'adaptive' (default) | 'extended' (expensive fallback)

        Returns:
            CompactionResult with compacted conversation + metadata
        """
        if not conversation:
            return CompactionResult(conversation=[], metadata={'tokens_before': 0, 'tokens_after': 0, 'reduction_pct': 0.0, 'strategy_used': strategy, 'thinking_used': False, 'thinking_tokens': 0, 'model': self.model, 'timestamp': datetime.now(timezone.utc).isoformat(), 'success': True}, is_sufficient=True)
        limit = token_budget or self.token_limit
        compaction_prompt = self._build_compaction_prompt(conversation, limit)
        if strategy == 'adaptive':
            result = self._call_with_adaptive_thinking(compaction_prompt, conversation, limit)
        elif strategy == 'extended' and self.enable_extended:
            result = self._call_with_extended_thinking(compaction_prompt, conversation, limit)
        else:
            result = self._fallback_heuristic_compact(conversation, limit)
        log.info(f'[gpt5-compactor] compact complete: {result.metadata['tokens_before']:,} → {result.metadata['tokens_after']:,} tokens ({result.metadata['reduction_pct']:.1f}% reduction, strategy={result.metadata['strategy_used']}, thinking_tokens={result.metadata['thinking_tokens']}, success={result.metadata['success']})')
        return result

    def _build_compaction_prompt(self, conversation: list[dict], token_limit: int) -> str:
        """Build a prompt instructing the model to compact the conversation."""
        current_tokens = self._estimate_tokens(json.dumps(conversation))
        target_tokens = int(token_limit * (self.threshold_pct / 100.0))
        prompt = f'You are a context compaction specialist. Analyze this conversation and produce a compacted version.\n\nCurrent token usage: ~{current_tokens:,} tokens\nToken limit: {token_limit:,} tokens\nTarget usage: ~{target_tokens:,} tokens (below {self.threshold_pct}% threshold)\n\nTask:\n1. Identify key information that MUST be preserved (user intent, critical context, decisions)\n2. Identify redundant or tool-output content that can be safely removed/summarized\n3. Keep system messages and recent turns intact\n4. Clean up verbose tool outputs, keeping only structured results\n5. Provide a JSON array of compacted messages\n\nOutput format: valid JSON array only, no explanation.\n\nConversation to compact:\n{json.dumps(conversation, indent=2)[:5000]}...\n\nCompacted conversation (valid JSON array):'
        return prompt

    def _estimate_tokens(self, text: str) -> int:
        """Quick token estimation (1 token ≈ 3-4 chars for general text)."""
        if not text:
            return 0
        return max(1, len(text) // 3)

    def _call_with_adaptive_thinking(self, prompt: str, conversation: list[dict], token_limit: int) -> CompactionResult:
        """
        Call gpt-5-mini with adaptive thinking (fast, free).
        
        Adaptive thinking allows gpt-5-mini to reason through problems efficiently
        without the cost of extended thinking. Perfect for routine compaction tasks.
        """
        try:
            from .worker import WorkerAgent
            agent = WorkerAgent('code_agent', complexity='simple')
            task = {'id': 'compaction_adaptive', 'agent': 'code_agent', 'type': 'reasoning', 'description': 'Compact conversation using adaptive thinking', 'request': prompt, 'complexity': 'simple', 'model': self.model, 'thinking': {'adaptive': True}}
            log.info(f'[gpt5-compactor] calling {self.model} with adaptive thinking...')
            result = agent.run(task, context={})
            if result.get('status') == 'completed':
                compacted = self._parse_compaction_response(result.get('output', ''), conversation)
                tokens_after = self._estimate_tokens(json.dumps(compacted))
                tokens_before = self._estimate_tokens(json.dumps(conversation))
                reduction = (tokens_before - tokens_after) / max(1, tokens_before) * 100
                return CompactionResult(conversation=compacted, metadata={'tokens_before': tokens_before, 'tokens_after': tokens_after, 'reduction_pct': reduction, 'strategy_used': 'adaptive', 'thinking_used': True, 'thinking_tokens': 0, 'model': self.model, 'timestamp': datetime.now(timezone.utc).isoformat(), 'success': True}, is_sufficient=tokens_after / token_limit < self.threshold_pct / 100.0)
            else:
                log.warning(f'[gpt5-compactor] adaptive thinking failed: {result.get('output')}')
                return self._fallback_heuristic_compact(conversation, token_limit)
        except Exception as e:
            log.warning(f'[gpt5-compactor] adaptive thinking call failed: {e}')
            return self._fallback_heuristic_compact(conversation, token_limit)

    def _call_with_extended_thinking(self, prompt: str, conversation: list[dict], token_limit: int) -> CompactionResult:
        """
        Call gpt-5-mini with extended thinking (slower, but thorough).
        
        Extended thinking tokens are automatically excluded from previous turns
        when passed back as conversation history, so this is still token-efficient.
        Only used when adaptive thinking is insufficient.
        """
        try:
            from .worker import WorkerAgent
            agent = WorkerAgent('code_agent', complexity='medium')
            task = {'id': 'compaction_extended', 'agent': 'code_agent', 'type': 'reasoning', 'description': 'Compact conversation using extended thinking', 'request': prompt, 'complexity': 'medium', 'model': self.model, 'thinking': {'adaptive': False, 'extended': True, 'max_thinking_tokens': 10000}}
            log.info(f'[gpt5-compactor] calling {self.model} with extended thinking (fallback)...')
            result = agent.run(task, context={})
            if result.get('status') == 'completed':
                compacted = self._parse_compaction_response(result.get('output', ''), conversation)
                tokens_after = self._estimate_tokens(json.dumps(compacted))
                tokens_before = self._estimate_tokens(json.dumps(conversation))
                reduction = (tokens_before - tokens_after) / max(1, tokens_before) * 100
                return CompactionResult(conversation=compacted, metadata={'tokens_before': tokens_before, 'tokens_after': tokens_after, 'reduction_pct': reduction, 'strategy_used': 'extended', 'thinking_used': True, 'thinking_tokens': 10000, 'model': self.model, 'timestamp': datetime.now(timezone.utc).isoformat(), 'success': True}, is_sufficient=tokens_after / token_limit < self.threshold_pct / 100.0)
            else:
                log.warning(f'[gpt5-compactor] extended thinking failed: {result.get('output')}')
                return self._fallback_heuristic_compact(conversation, token_limit)
        except Exception as e:
            log.warning(f'[gpt5-compactor] extended thinking call failed: {e}')
            return self._fallback_heuristic_compact(conversation, token_limit)

    def _parse_compaction_response(self, response_text: str, original: list[dict]) -> list[dict]:
        """
        Parse model's compaction response.
        
        Tries to extract JSON array; falls back to heuristic if parsing fails.
        """
        try:
            response_text = response_text.strip()
            if response_text.startswith('['):
                compacted = json.loads(response_text)
                if isinstance(compacted, list):
                    log.info(f'[gpt5-compactor] parsed compacted response: {len(original)} → {len(compacted)} messages')
                    return compacted
        except json.JSONDecodeError:
            pass
        log.warning('[gpt5-compactor] failed to parse model response; using heuristic fallback')
        return self._fallback_heuristic_compact(original, self.token_limit).conversation

    def _fallback_heuristic_compact(self, conversation: list[dict], token_limit: int) -> CompactionResult:
        """
        Fallback heuristic compaction (no model call).
        
        Used when model is unavailable or calls fail.
        Implements the same strategies as context_compactor.py.
        """
        from .context_compactor import ContextCompactor
        heuristic = ContextCompactor(token_limit=token_limit)
        compacted = heuristic.compact(conversation, strategy='combined', keep_last_n_turns=10)
        tokens_before = heuristic.estimate_tokens(conversation).tokens_used
        tokens_after = heuristic.estimate_tokens(compacted).tokens_used
        reduction = (tokens_before - tokens_after) / max(1, tokens_before) * 100
        return CompactionResult(conversation=compacted, metadata={'tokens_before': tokens_before, 'tokens_after': tokens_after, 'reduction_pct': reduction, 'strategy_used': 'heuristic_fallback', 'thinking_used': False, 'thinking_tokens': 0, 'model': 'heuristic', 'timestamp': datetime.now(timezone.utc).isoformat(), 'success': True}, is_sufficient=tokens_after / token_limit < self.threshold_pct / 100.0)

    def get_compaction_metadata(self) -> dict:
        """Return current compactor configuration as metadata."""
        return {'model': self.model, 'backend': self.backend, 'token_limit': self.token_limit, 'threshold_pct': self.threshold_pct, 'enable_extended_thinking': self.enable_extended, 'description': 'GPT-5-Mini Adaptive Thinking Compactor (free tier)'}

class CompactionMetadata__from_archived_1007(TypedDict):
    """Metadata about compaction operation."""
    tokens_before: int
    tokens_after: int
    reduction_pct: float
    strategy_used: str
    thinking_used: bool
    thinking_tokens: int
    model: str
    timestamp: str
    success: bool

@dataclass
class CompactionResult__from_archived_1007:
    """Result of compaction operation."""
    conversation: list[dict]
    metadata: CompactionMetadata
    is_sufficient: bool

class GPT5AdaptiveCompactor__from_archived_1007:
    """
    Model-based context compactor using gpt-5-mini with adaptive thinking.
    
    Integrates with AutoBeast's model selection (via Copilot SDK / worker.py).
    Uses free gpt-5-mini tier; escalates to extended thinking only when needed.
    """

    def __init__(self, model: str='gpt-5-mini', token_limit: int=200000, compaction_threshold_pct: float=85.0, enable_extended_thinking: bool=True):
        """
        Args:
            model: Model to use (default: gpt-5-mini, free)
            token_limit: Maximum context window
            compaction_threshold_pct: Target threshold after compaction
            enable_extended_thinking: Allow fallback to extended thinking if adaptive insufficient
        """
        self.model = model
        self.token_limit = token_limit
        self.threshold_pct = compaction_threshold_pct
        self.enable_extended = enable_extended_thinking
        self.backend = BEAST_BACKEND
        log.info(f'[gpt5-compactor] initialized: model={model}, backend={self.backend}, token_limit={token_limit:,}, threshold={compaction_threshold_pct}%')

    def compact(self, conversation: list[dict], token_budget: Optional[int]=None, strategy: str='adaptive') -> CompactionResult:
        """
        Compact conversation using gpt-5-mini with adaptive thinking.

        Args:
            conversation: Message history
            token_budget: Optional override for token limit (uses self.token_limit if None)
            strategy: 'adaptive' (default) | 'extended' (expensive fallback)

        Returns:
            CompactionResult with compacted conversation + metadata
        """
        if not conversation:
            return CompactionResult(conversation=[], metadata={'tokens_before': 0, 'tokens_after': 0, 'reduction_pct': 0.0, 'strategy_used': strategy, 'thinking_used': False, 'thinking_tokens': 0, 'model': self.model, 'timestamp': datetime.now(timezone.utc).isoformat(), 'success': True}, is_sufficient=True)
        limit = token_budget or self.token_limit
        compaction_prompt = self._build_compaction_prompt(conversation, limit)
        if strategy == 'adaptive':
            result = self._call_with_adaptive_thinking(compaction_prompt, conversation, limit)
        elif strategy == 'extended' and self.enable_extended:
            result = self._call_with_extended_thinking(compaction_prompt, conversation, limit)
        else:
            result = self._fallback_heuristic_compact(conversation, limit)
        log.info(f'[gpt5-compactor] compact complete: {result.metadata['tokens_before']:,} → {result.metadata['tokens_after']:,} tokens ({result.metadata['reduction_pct']:.1f}% reduction, strategy={result.metadata['strategy_used']}, thinking_tokens={result.metadata['thinking_tokens']}, success={result.metadata['success']})')
        return result

    def _build_compaction_prompt(self, conversation: list[dict], token_limit: int) -> str:
        """Build a prompt instructing the model to compact the conversation."""
        current_tokens = self._estimate_tokens(json.dumps(conversation))
        target_tokens = int(token_limit * (self.threshold_pct / 100.0))
        prompt = f'You are a context compaction specialist. Analyze this conversation and produce a compacted version.\n\nCurrent token usage: ~{current_tokens:,} tokens\nToken limit: {token_limit:,} tokens\nTarget usage: ~{target_tokens:,} tokens (below {self.threshold_pct}% threshold)\n\nTask:\n1. Identify key information that MUST be preserved (user intent, critical context, decisions)\n2. Identify redundant or tool-output content that can be safely removed/summarized\n3. Keep system messages and recent turns intact\n4. Clean up verbose tool outputs, keeping only structured results\n5. Provide a JSON array of compacted messages\n\nOutput format: valid JSON array only, no explanation.\n\nConversation to compact:\n{json.dumps(conversation, indent=2)[:5000]}...\n\nCompacted conversation (valid JSON array):'
        return prompt

    def _estimate_tokens(self, text: str) -> int:
        """Quick token estimation (1 token ≈ 3-4 chars for general text)."""
        if not text:
            return 0
        return max(1, len(text) // 3)

    def _call_with_adaptive_thinking(self, prompt: str, conversation: list[dict], token_limit: int) -> CompactionResult:
        """
        Call gpt-5-mini with adaptive thinking (fast, free).
        
        Adaptive thinking allows gpt-5-mini to reason through problems efficiently
        without the cost of extended thinking. Perfect for routine compaction tasks.
        """
        try:
            from .worker import WorkerAgent
            agent = WorkerAgent('code_agent', complexity='simple')
            task = {'id': 'compaction_adaptive', 'agent': 'code_agent', 'type': 'reasoning', 'description': 'Compact conversation using adaptive thinking', 'request': prompt, 'complexity': 'simple', 'model': self.model, 'thinking': {'adaptive': True}}
            log.info(f'[gpt5-compactor] calling {self.model} with adaptive thinking...')
            result = agent.run(task, context={})
            if result.get('status') == 'completed':
                compacted = self._parse_compaction_response(result.get('output', ''), conversation)
                tokens_after = self._estimate_tokens(json.dumps(compacted))
                tokens_before = self._estimate_tokens(json.dumps(conversation))
                reduction = (tokens_before - tokens_after) / max(1, tokens_before) * 100
                return CompactionResult(conversation=compacted, metadata={'tokens_before': tokens_before, 'tokens_after': tokens_after, 'reduction_pct': reduction, 'strategy_used': 'adaptive', 'thinking_used': True, 'thinking_tokens': 0, 'model': self.model, 'timestamp': datetime.now(timezone.utc).isoformat(), 'success': True}, is_sufficient=tokens_after / token_limit < self.threshold_pct / 100.0)
            else:
                log.warning(f'[gpt5-compactor] adaptive thinking failed: {result.get('output')}')
                return self._fallback_heuristic_compact(conversation, token_limit)
        except Exception as e:
            log.warning(f'[gpt5-compactor] adaptive thinking call failed: {e}')
            return self._fallback_heuristic_compact(conversation, token_limit)

    def _call_with_extended_thinking(self, prompt: str, conversation: list[dict], token_limit: int) -> CompactionResult:
        """
        Call gpt-5-mini with extended thinking (slower, but thorough).
        
        Extended thinking tokens are automatically excluded from previous turns
        when passed back as conversation history, so this is still token-efficient.
        Only used when adaptive thinking is insufficient.
        """
        try:
            from .worker import WorkerAgent
            agent = WorkerAgent('code_agent', complexity='medium')
            task = {'id': 'compaction_extended', 'agent': 'code_agent', 'type': 'reasoning', 'description': 'Compact conversation using extended thinking', 'request': prompt, 'complexity': 'medium', 'model': self.model, 'thinking': {'adaptive': False, 'extended': True, 'max_thinking_tokens': 10000}}
            log.info(f'[gpt5-compactor] calling {self.model} with extended thinking (fallback)...')
            result = agent.run(task, context={})
            if result.get('status') == 'completed':
                compacted = self._parse_compaction_response(result.get('output', ''), conversation)
                tokens_after = self._estimate_tokens(json.dumps(compacted))
                tokens_before = self._estimate_tokens(json.dumps(conversation))
                reduction = (tokens_before - tokens_after) / max(1, tokens_before) * 100
                return CompactionResult(conversation=compacted, metadata={'tokens_before': tokens_before, 'tokens_after': tokens_after, 'reduction_pct': reduction, 'strategy_used': 'extended', 'thinking_used': True, 'thinking_tokens': 10000, 'model': self.model, 'timestamp': datetime.now(timezone.utc).isoformat(), 'success': True}, is_sufficient=tokens_after / token_limit < self.threshold_pct / 100.0)
            else:
                log.warning(f'[gpt5-compactor] extended thinking failed: {result.get('output')}')
                return self._fallback_heuristic_compact(conversation, token_limit)
        except Exception as e:
            log.warning(f'[gpt5-compactor] extended thinking call failed: {e}')
            return self._fallback_heuristic_compact(conversation, token_limit)

    def _parse_compaction_response(self, response_text: str, original: list[dict]) -> list[dict]:
        """
        Parse model's compaction response.
        
        Tries to extract JSON array; falls back to heuristic if parsing fails.
        """
        try:
            response_text = response_text.strip()
            if response_text.startswith('['):
                compacted = json.loads(response_text)
                if isinstance(compacted, list):
                    log.info(f'[gpt5-compactor] parsed compacted response: {len(original)} → {len(compacted)} messages')
                    return compacted
        except json.JSONDecodeError:
            pass
        log.warning('[gpt5-compactor] failed to parse model response; using heuristic fallback')
        return self._fallback_heuristic_compact(original, self.token_limit).conversation

    def _fallback_heuristic_compact(self, conversation: list[dict], token_limit: int) -> CompactionResult:
        """
        Fallback heuristic compaction (no model call).
        
        Used when model is unavailable or calls fail.
        Implements the same strategies as context_compactor.py.
        """
        from .context_compactor import ContextCompactor
        heuristic = ContextCompactor(token_limit=token_limit)
        compacted = heuristic.compact(conversation, strategy='combined', keep_last_n_turns=10)
        tokens_before = heuristic.estimate_tokens(conversation).tokens_used
        tokens_after = heuristic.estimate_tokens(compacted).tokens_used
        reduction = (tokens_before - tokens_after) / max(1, tokens_before) * 100
        return CompactionResult(conversation=compacted, metadata={'tokens_before': tokens_before, 'tokens_after': tokens_after, 'reduction_pct': reduction, 'strategy_used': 'heuristic_fallback', 'thinking_used': False, 'thinking_tokens': 0, 'model': 'heuristic', 'timestamp': datetime.now(timezone.utc).isoformat(), 'success': True}, is_sufficient=tokens_after / token_limit < self.threshold_pct / 100.0)

    def get_compaction_metadata(self) -> dict:
        """Return current compactor configuration as metadata."""
        return {'model': self.model, 'backend': self.backend, 'token_limit': self.token_limit, 'threshold_pct': self.threshold_pct, 'enable_extended_thinking': self.enable_extended, 'description': 'GPT-5-Mini Adaptive Thinking Compactor (free tier)'}

class CompactionMetadata__from_archived_8789(TypedDict):
    """Metadata about compaction operation."""
    tokens_before: int
    tokens_after: int
    reduction_pct: float
    strategy_used: str
    thinking_used: bool
    thinking_tokens: int
    model: str
    timestamp: str
    success: bool

@dataclass
class CompactionResult__from_archived_8789:
    """Result of compaction operation."""
    conversation: list[dict]
    metadata: CompactionMetadata
    is_sufficient: bool

class GPT5AdaptiveCompactor__from_archived_8789:
    """
    Model-based context compactor using gpt-5-mini with adaptive thinking.
    
    Integrates with AutoBeast's model selection (via Copilot SDK / worker.py).
    Uses free gpt-5-mini tier; escalates to extended thinking only when needed.
    """

    def __init__(self, model: str='gpt-5-mini', token_limit: int=200000, compaction_threshold_pct: float=85.0, enable_extended_thinking: bool=True):
        """
        Args:
            model: Model to use (default: gpt-5-mini, free)
            token_limit: Maximum context window
            compaction_threshold_pct: Target threshold after compaction
            enable_extended_thinking: Allow fallback to extended thinking if adaptive insufficient
        """
        self.model = model
        self.token_limit = token_limit
        self.threshold_pct = compaction_threshold_pct
        self.enable_extended = enable_extended_thinking
        self.backend = BEAST_BACKEND
        log.info(f'[gpt5-compactor] initialized: model={model}, backend={self.backend}, token_limit={token_limit:,}, threshold={compaction_threshold_pct}%')

    def compact(self, conversation: list[dict], token_budget: Optional[int]=None, strategy: str='adaptive') -> CompactionResult:
        """
        Compact conversation using gpt-5-mini with adaptive thinking.

        Args:
            conversation: Message history
            token_budget: Optional override for token limit (uses self.token_limit if None)
            strategy: 'adaptive' (default) | 'extended' (expensive fallback)

        Returns:
            CompactionResult with compacted conversation + metadata
        """
        if not conversation:
            return CompactionResult(conversation=[], metadata={'tokens_before': 0, 'tokens_after': 0, 'reduction_pct': 0.0, 'strategy_used': strategy, 'thinking_used': False, 'thinking_tokens': 0, 'model': self.model, 'timestamp': datetime.now(timezone.utc).isoformat(), 'success': True}, is_sufficient=True)
        limit = token_budget or self.token_limit
        compaction_prompt = self._build_compaction_prompt(conversation, limit)
        if strategy == 'adaptive':
            result = self._call_with_adaptive_thinking(compaction_prompt, conversation, limit)
        elif strategy == 'extended' and self.enable_extended:
            result = self._call_with_extended_thinking(compaction_prompt, conversation, limit)
        else:
            result = self._fallback_heuristic_compact(conversation, limit)
        log.info(f'[gpt5-compactor] compact complete: {result.metadata['tokens_before']:,} → {result.metadata['tokens_after']:,} tokens ({result.metadata['reduction_pct']:.1f}% reduction, strategy={result.metadata['strategy_used']}, thinking_tokens={result.metadata['thinking_tokens']}, success={result.metadata['success']})')
        return result

    def _build_compaction_prompt(self, conversation: list[dict], token_limit: int) -> str:
        """Build a prompt instructing the model to compact the conversation."""
        current_tokens = self._estimate_tokens(json.dumps(conversation))
        target_tokens = int(token_limit * (self.threshold_pct / 100.0))
        prompt = f'You are a context compaction specialist. Analyze this conversation and produce a compacted version.\n\nCurrent token usage: ~{current_tokens:,} tokens\nToken limit: {token_limit:,} tokens\nTarget usage: ~{target_tokens:,} tokens (below {self.threshold_pct}% threshold)\n\nTask:\n1. Identify key information that MUST be preserved (user intent, critical context, decisions)\n2. Identify redundant or tool-output content that can be safely removed/summarized\n3. Keep system messages and recent turns intact\n4. Clean up verbose tool outputs, keeping only structured results\n5. Provide a JSON array of compacted messages\n\nOutput format: valid JSON array only, no explanation.\n\nConversation to compact:\n{json.dumps(conversation, indent=2)[:5000]}...\n\nCompacted conversation (valid JSON array):'
        return prompt

    def _estimate_tokens(self, text: str) -> int:
        """Quick token estimation (1 token ≈ 3-4 chars for general text)."""
        if not text:
            return 0
        return max(1, len(text) // 3)

    def _call_with_adaptive_thinking(self, prompt: str, conversation: list[dict], token_limit: int) -> CompactionResult:
        """
        Call gpt-5-mini with adaptive thinking (fast, free).
        
        Adaptive thinking allows gpt-5-mini to reason through problems efficiently
        without the cost of extended thinking. Perfect for routine compaction tasks.
        """
        try:
            from .worker import WorkerAgent
            agent = WorkerAgent('code_agent', complexity='simple')
            task = {'id': 'compaction_adaptive', 'agent': 'code_agent', 'type': 'reasoning', 'description': 'Compact conversation using adaptive thinking', 'request': prompt, 'complexity': 'simple', 'model': self.model, 'thinking': {'adaptive': True}}
            log.info(f'[gpt5-compactor] calling {self.model} with adaptive thinking...')
            result = agent.run(task, context={})
            if result.get('status') == 'completed':
                compacted = self._parse_compaction_response(result.get('output', ''), conversation)
                tokens_after = self._estimate_tokens(json.dumps(compacted))
                tokens_before = self._estimate_tokens(json.dumps(conversation))
                reduction = (tokens_before - tokens_after) / max(1, tokens_before) * 100
                return CompactionResult(conversation=compacted, metadata={'tokens_before': tokens_before, 'tokens_after': tokens_after, 'reduction_pct': reduction, 'strategy_used': 'adaptive', 'thinking_used': True, 'thinking_tokens': 0, 'model': self.model, 'timestamp': datetime.now(timezone.utc).isoformat(), 'success': True}, is_sufficient=tokens_after / token_limit < self.threshold_pct / 100.0)
            else:
                log.warning(f'[gpt5-compactor] adaptive thinking failed: {result.get('output')}')
                return self._fallback_heuristic_compact(conversation, token_limit)
        except Exception as e:
            log.warning(f'[gpt5-compactor] adaptive thinking call failed: {e}')
            return self._fallback_heuristic_compact(conversation, token_limit)

    def _call_with_extended_thinking(self, prompt: str, conversation: list[dict], token_limit: int) -> CompactionResult:
        """
        Call gpt-5-mini with extended thinking (slower, but thorough).
        
        Extended thinking tokens are automatically excluded from previous turns
        when passed back as conversation history, so this is still token-efficient.
        Only used when adaptive thinking is insufficient.
        """
        try:
            from .worker import WorkerAgent
            agent = WorkerAgent('code_agent', complexity='medium')
            task = {'id': 'compaction_extended', 'agent': 'code_agent', 'type': 'reasoning', 'description': 'Compact conversation using extended thinking', 'request': prompt, 'complexity': 'medium', 'model': self.model, 'thinking': {'adaptive': False, 'extended': True, 'max_thinking_tokens': 10000}}
            log.info(f'[gpt5-compactor] calling {self.model} with extended thinking (fallback)...')
            result = agent.run(task, context={})
            if result.get('status') == 'completed':
                compacted = self._parse_compaction_response(result.get('output', ''), conversation)
                tokens_after = self._estimate_tokens(json.dumps(compacted))
                tokens_before = self._estimate_tokens(json.dumps(conversation))
                reduction = (tokens_before - tokens_after) / max(1, tokens_before) * 100
                return CompactionResult(conversation=compacted, metadata={'tokens_before': tokens_before, 'tokens_after': tokens_after, 'reduction_pct': reduction, 'strategy_used': 'extended', 'thinking_used': True, 'thinking_tokens': 10000, 'model': self.model, 'timestamp': datetime.now(timezone.utc).isoformat(), 'success': True}, is_sufficient=tokens_after / token_limit < self.threshold_pct / 100.0)
            else:
                log.warning(f'[gpt5-compactor] extended thinking failed: {result.get('output')}')
                return self._fallback_heuristic_compact(conversation, token_limit)
        except Exception as e:
            log.warning(f'[gpt5-compactor] extended thinking call failed: {e}')
            return self._fallback_heuristic_compact(conversation, token_limit)

    def _parse_compaction_response(self, response_text: str, original: list[dict]) -> list[dict]:
        """
        Parse model's compaction response.
        
        Tries to extract JSON array; falls back to heuristic if parsing fails.
        """
        try:
            response_text = response_text.strip()
            if response_text.startswith('['):
                compacted = json.loads(response_text)
                if isinstance(compacted, list):
                    log.info(f'[gpt5-compactor] parsed compacted response: {len(original)} → {len(compacted)} messages')
                    return compacted
        except json.JSONDecodeError:
            pass
        log.warning('[gpt5-compactor] failed to parse model response; using heuristic fallback')
        return self._fallback_heuristic_compact(original, self.token_limit).conversation

    def _fallback_heuristic_compact(self, conversation: list[dict], token_limit: int) -> CompactionResult:
        """
        Fallback heuristic compaction (no model call).
        
        Used when model is unavailable or calls fail.
        Implements the same strategies as context_compactor.py.
        """
        from .context_compactor import ContextCompactor
        heuristic = ContextCompactor(token_limit=token_limit)
        compacted = heuristic.compact(conversation, strategy='combined', keep_last_n_turns=10)
        tokens_before = heuristic.estimate_tokens(conversation).tokens_used
        tokens_after = heuristic.estimate_tokens(compacted).tokens_used
        reduction = (tokens_before - tokens_after) / max(1, tokens_before) * 100
        return CompactionResult(conversation=compacted, metadata={'tokens_before': tokens_before, 'tokens_after': tokens_after, 'reduction_pct': reduction, 'strategy_used': 'heuristic_fallback', 'thinking_used': False, 'thinking_tokens': 0, 'model': 'heuristic', 'timestamp': datetime.now(timezone.utc).isoformat(), 'success': True}, is_sufficient=tokens_after / token_limit < self.threshold_pct / 100.0)

    def get_compaction_metadata(self) -> dict:
        """Return current compactor configuration as metadata."""
        return {'model': self.model, 'backend': self.backend, 'token_limit': self.token_limit, 'threshold_pct': self.threshold_pct, 'enable_extended_thinking': self.enable_extended, 'description': 'GPT-5-Mini Adaptive Thinking Compactor (free tier)'}

# === MERGED FROM C:\Users\AmitMohanty\.vscode\brain\.agent\tmp\moved_duplicates\20260328T133840Z\gpt5_adaptive_compactor.py ===
# """
# gpt5_adaptive_compactor.py — GPT-5-Mini Adaptive Thinking Context Compactor
# 
# Uses gpt-5-mini (free tier) with adaptive thinking for intelligent context compaction.
# Implements Claude's context management architecture:
#   - Thinking blocks automatically excluded from previous turns
#   - Extended thinking tokens billed as output tokens only (fallback)
#   - Effective context window = (input_tokens - previous_thinking_tokens) + current_turn_tokens
#   - Token-efficient extensive reasoning without waste
# 
# References:
#   - platform.claude.com/docs/build-with-claude/token-counting
#   - platform.claude.com/docs/build-with-claude/context-windows
#   - platform.claude.com/docs/build-with-claude/compaction
#   - platform.claude.com/docs/build-with-claude/context-editing
#   - platform.claude.com/docs/build-with-claude/prompt-caching
# 
# Strategy:
#   1. Adaptive thinking (default): Fast, free, suitable for most compaction scenarios
#   2. Extended thinking (fallback): Only when adaptive is insufficient or task is complex
#   3. Thinking exclusion: Leverage automatic thinking block filtering in API
# 
# USAGE:
#   compactor = GPT5AdaptiveCompactor()
#   compacted = compactor.compact(
#       conversation=history,
#       token_budget=200_000,
#       strategy='adaptive',  # or 'extended'
#   )
#   metadata = compactor.get_compaction_metadata()
# """
# 
# import logging
# import json
# import os
# from pathlib import Path
# from typing import Optional, TypedDict
# from dataclasses import dataclass, asdict
# from datetime import datetime, timezone
# 
# log = logging.getLogger(__name__)
# 
# # Environment variable to control backend (default: copilot-sdk, which uses gpt-5-mini)
# BEAST_BACKEND = os.environ.get("BEAST_BACKEND", "copilot-sdk")
# 
# 
# class CompactionMetadata(TypedDict):
#     """Metadata about compaction operation."""
#     tokens_before: int
#     tokens_after: int
#     reduction_pct: float
#     strategy_used: str  # 'adaptive' | 'extended'
#     thinking_used: bool
#     thinking_tokens: int
#     model: str
#     timestamp: str
#     success: bool
# 
# 
# @dataclass
# class CompactionResult:
#     """Result of compaction operation."""
#     conversation: list[dict]
#     metadata: CompactionMetadata
#     is_sufficient: bool  # True if reduced below threshold
# 
# 
# class GPT5AdaptiveCompactor:
#     """
#     Model-based context compactor using gpt-5-mini with adaptive thinking.
#     
#     Integrates with AutoBeast's model selection (via Copilot SDK / worker.py).
#     Uses free gpt-5-mini tier; escalates to extended thinking only when needed.
#     """
# 
#     def __init__(
#         self,
#         model: str = "gpt-5-mini",
#         token_limit: int = 200_000,
#         compaction_threshold_pct: float = 85.0,
#         enable_extended_thinking: bool = True,
#     ):
#         """
#         Args:
#             model: Model to use (default: gpt-5-mini, free)
#             token_limit: Maximum context window
#             compaction_threshold_pct: Target threshold after compaction
#             enable_extended_thinking: Allow fallback to extended thinking if adaptive insufficient
#         """
#         self.model = model
#         self.token_limit = token_limit
#         self.threshold_pct = compaction_threshold_pct
#         self.enable_extended = enable_extended_thinking
#         self.backend = BEAST_BACKEND
#         
#         log.info(
#             f"[gpt5-compactor] initialized: model={model}, backend={self.backend}, "
#             f"token_limit={token_limit:,}, threshold={compaction_threshold_pct}%"
#         )
# 
#     def compact(
#         self,
#         conversation: list[dict],
#         token_budget: Optional[int] = None,
#         strategy: str = "adaptive",
#     ) -> CompactionResult:
#         """
#         Compact conversation using gpt-5-mini with adaptive thinking.
# 
#         Args:
#             conversation: Message history
#             token_budget: Optional override for token limit (uses self.token_limit if None)
#             strategy: 'adaptive' (default) | 'extended' (expensive fallback)
# 
#         Returns:
#             CompactionResult with compacted conversation + metadata
#         """
#         if not conversation:
#             return CompactionResult(
#                 conversation=[],
#                 metadata={
#                     "tokens_before": 0,
#                     "tokens_after": 0,
#                     "reduction_pct": 0.0,
#                     "strategy_used": strategy,
#                     "thinking_used": False,
#                     "thinking_tokens": 0,
#                     "model": self.model,
#                     "timestamp": datetime.now(timezone.utc).isoformat(),
#                     "success": True,
#                 },
#                 is_sufficient=True,
#             )
# 
#         limit = token_budget or self.token_limit
# 
#         # Build compaction prompt for the model
#         compaction_prompt = self._build_compaction_prompt(conversation, limit)
# 
#         # Call model with adaptive thinking (free) → fallback to extended if needed
#         if strategy == "adaptive":
#             result = self._call_with_adaptive_thinking(compaction_prompt, conversation, limit)
#         elif strategy == "extended" and self.enable_extended:
#             result = self._call_with_extended_thinking(compaction_prompt, conversation, limit)
#         else:
#             # Fallback to heuristic if model call unavailable
#             result = self._fallback_heuristic_compact(conversation, limit)
# 
#         log.info(
#             f"[gpt5-compactor] compact complete: {result.metadata['tokens_before']:,} → "
#             f"{result.metadata['tokens_after']:,} tokens "
#             f"({result.metadata['reduction_pct']:.1f}% reduction, "
#             f"strategy={result.metadata['strategy_used']}, "
#             f"thinking_tokens={result.metadata['thinking_tokens']}, "
#             f"success={result.metadata['success']})"
#         )
# 
#         return result
# 
#     def _build_compaction_prompt(self, conversation: list[dict], token_limit: int) -> str:
#         """Build a prompt instructing the model to compact the conversation."""
#         current_tokens = self._estimate_tokens(json.dumps(conversation))
#         target_tokens = int(token_limit * (self.threshold_pct / 100.0))
# 
#         prompt = f"""You are a context compaction specialist. Analyze this conversation and produce a compacted version.
# 
# Current token usage: ~{current_tokens:,} tokens
# Token limit: {token_limit:,} tokens
# Target usage: ~{target_tokens:,} tokens (below {self.threshold_pct}% threshold)
# 
# Task:
# 1. Identify key information that MUST be preserved (user intent, critical context, decisions)
# 2. Identify redundant or tool-output content that can be safely removed/summarized
# 3. Keep system messages and recent turns intact
# 4. Clean up verbose tool outputs, keeping only structured results
# 5. Provide a JSON array of compacted messages
# 
# Output format: valid JSON array only, no explanation.
# 
# Conversation to compact:
# {json.dumps(conversation, indent=2)[:5000]}...
# 
# Compacted conversation (valid JSON array):"""
#         return prompt
# 
#     def _estimate_tokens(self, text: str) -> int:
#         """Quick token estimation (1 token ≈ 3-4 chars for general text)."""
#         if not text:
#             return 0
#         return max(1, len(text) // 3)
# 
#     def _call_with_adaptive_thinking(
#         self, prompt: str, conversation: list[dict], token_limit: int
#     ) -> CompactionResult:
#         """
#         Call gpt-5-mini with adaptive thinking (fast, free).
#         
#         Adaptive thinking allows gpt-5-mini to reason through problems efficiently
#         without the cost of extended thinking. Perfect for routine compaction tasks.
#         """
#         try:
#             # Try to use worker.py's model invocation if available
#             from .worker import WorkerAgent
#             
#             agent = WorkerAgent("code_agent", complexity="simple")
#             
#             # Build a minimal task for the model
#             task = {
#                 "id": "compaction_adaptive",
#                 "agent": "code_agent",
#                 "type": "reasoning",
#                 "description": "Compact conversation using adaptive thinking",
#                 "request": prompt,
#                 "complexity": "simple",
#                 "model": self.model,
#                 "thinking": {"adaptive": True},  # Enable adaptive thinking
#             }
# 
#             log.info(f"[gpt5-compactor] calling {self.model} with adaptive thinking...")
#             result = agent.run(task, context={})
# 
#             if result.get("status") == "completed":
#                 compacted = self._parse_compaction_response(result.get("output", ""), conversation)
#                 tokens_after = self._estimate_tokens(json.dumps(compacted))
#                 tokens_before = self._estimate_tokens(json.dumps(conversation))
#                 reduction = ((tokens_before - tokens_after) / max(1, tokens_before)) * 100
# 
#                 return CompactionResult(
#                     conversation=compacted,
#                     metadata={
#                         "tokens_before": tokens_before,
#                         "tokens_after": tokens_after,
#                         "reduction_pct": reduction,
#                         "strategy_used": "adaptive",
#                         "thinking_used": True,
#                         "thinking_tokens": 0,  # Not tracked in response
#                         "model": self.model,
#                         "timestamp": datetime.now(timezone.utc).isoformat(),
#                         "success": True,
#                     },
#                     is_sufficient=(tokens_after / token_limit) < (self.threshold_pct / 100.0),
#                 )
#             else:
#                 log.warning(f"[gpt5-compactor] adaptive thinking failed: {result.get('output')}")
#                 return self._fallback_heuristic_compact(conversation, token_limit)
# 
#         except Exception as e:
#             log.warning(f"[gpt5-compactor] adaptive thinking call failed: {e}")
#             return self._fallback_heuristic_compact(conversation, token_limit)
# 
#     def _call_with_extended_thinking(
#         self, prompt: str, conversation: list[dict], token_limit: int
#     ) -> CompactionResult:
#         """
#         Call gpt-5-mini with extended thinking (slower, but thorough).
#         
#         Extended thinking tokens are automatically excluded from previous turns
#         when passed back as conversation history, so this is still token-efficient.
#         Only used when adaptive thinking is insufficient.
#         """
#         try:
#             from .worker import WorkerAgent
#             
#             agent = WorkerAgent("code_agent", complexity="medium")
#             
#             task = {
#                 "id": "compaction_extended",
#                 "agent": "code_agent",
#                 "type": "reasoning",
#                 "description": "Compact conversation using extended thinking",
#                 "request": prompt,
#                 "complexity": "medium",
#                 "model": self.model,
#                 "thinking": {
#                     "adaptive": False,
#                     "extended": True,  # Enable extended thinking (fallback)
#                     "max_thinking_tokens": 10_000,  # Limit to control cost
#                 },
#             }
# 
#             log.info(f"[gpt5-compactor] calling {self.model} with extended thinking (fallback)...")
#             result = agent.run(task, context={})
# 
#             if result.get("status") == "completed":
#                 compacted = self._parse_compaction_response(result.get("output", ""), conversation)
#                 tokens_after = self._estimate_tokens(json.dumps(compacted))
#                 tokens_before = self._estimate_tokens(json.dumps(conversation))
#                 reduction = ((tokens_before - tokens_after) / max(1, tokens_before)) * 100
# 
#                 return CompactionResult(
#                     conversation=compacted,
#                     metadata={
#                         "tokens_before": tokens_before,
#                         "tokens_after": tokens_after,
#                         "reduction_pct": reduction,
#                         "strategy_used": "extended",
#                         "thinking_used": True,
#                         "thinking_tokens": 10_000,  # Estimate
#                         "model": self.model,
#                         "timestamp": datetime.now(timezone.utc).isoformat(),
#                         "success": True,
#                     },
#                     is_sufficient=(tokens_after / token_limit) < (self.threshold_pct / 100.0),
#                 )
#             else:
#                 log.warning(f"[gpt5-compactor] extended thinking failed: {result.get('output')}")
#                 return self._fallback_heuristic_compact(conversation, token_limit)
# 
#         except Exception as e:
#             log.warning(f"[gpt5-compactor] extended thinking call failed: {e}")
#             return self._fallback_heuristic_compact(conversation, token_limit)
# 
#     def _parse_compaction_response(self, response_text: str, original: list[dict]) -> list[dict]:
#         """
#         Parse model's compaction response.
#         
#         Tries to extract JSON array; falls back to heuristic if parsing fails.
#         """
#         try:
#             # Try to find JSON array in response
#             response_text = response_text.strip()
#             if response_text.startswith("["):
#                 compacted = json.loads(response_text)
#                 if isinstance(compacted, list):
#                     log.info(f"[gpt5-compactor] parsed compacted response: {len(original)} → {len(compacted)} messages")
#                     return compacted
#         except json.JSONDecodeError:
#             pass
# 
#         log.warning("[gpt5-compactor] failed to parse model response; using heuristic fallback")
#         return self._fallback_heuristic_compact(original, self.token_limit).conversation
# 
#     def _fallback_heuristic_compact(
#         self, conversation: list[dict], token_limit: int
#     ) -> CompactionResult:
#         """
#         Fallback heuristic compaction (no model call).
#         
#         Used when model is unavailable or calls fail.
#         Implements the same strategies as context_compactor.py.
#         """
#         from .context_compactor import ContextCompactor
#         
#         heuristic = ContextCompactor(token_limit=token_limit)
#         compacted = heuristic.compact(conversation, strategy="combined", keep_last_n_turns=10)
#         
#         tokens_before = heuristic.estimate_tokens(conversation).tokens_used
#         tokens_after = heuristic.estimate_tokens(compacted).tokens_used
#         reduction = ((tokens_before - tokens_after) / max(1, tokens_before)) * 100
# 
#         return CompactionResult(
#             conversation=compacted,
#             metadata={
#                 "tokens_before": tokens_before,
#                 "tokens_after": tokens_after,
#                 "reduction_pct": reduction,
#                 "strategy_used": "heuristic_fallback",
#                 "thinking_used": False,
#                 "thinking_tokens": 0,
#                 "model": "heuristic",
#                 "timestamp": datetime.now(timezone.utc).isoformat(),
#                 "success": True,
#             },
#             is_sufficient=(tokens_after / token_limit) < (self.threshold_pct / 100.0),
#         )
# 
#     def get_compaction_metadata(self) -> dict:
#         """Return current compactor configuration as metadata."""
#         return {
#             "model": self.model,
#             "backend": self.backend,
#             "token_limit": self.token_limit,
#             "threshold_pct": self.threshold_pct,
#             "enable_extended_thinking": self.enable_extended,
#             "description": "GPT-5-Mini Adaptive Thinking Compactor (free tier)",
#         }


# === MERGED FROM C:\Users\AmitMohanty\.vscode\brain\.agent\tmp\moved_duplicates\20260328T133840Z\gpt5_adaptive_compactor.py.1 ===
# """
# gpt5_adaptive_compactor.py — GPT-5-Mini Adaptive Thinking Context Compactor
# 
# Uses gpt-5-mini (free tier) with adaptive thinking for intelligent context compaction.
# Implements Claude's context management architecture:
#   - Thinking blocks automatically excluded from previous turns
#   - Extended thinking tokens billed as output tokens only (fallback)
#   - Effective context window = (input_tokens - previous_thinking_tokens) + current_turn_tokens
#   - Token-efficient extensive reasoning without waste
# 
# References:
#   - platform.claude.com/docs/build-with-claude/token-counting
#   - platform.claude.com/docs/build-with-claude/context-windows
#   - platform.claude.com/docs/build-with-claude/compaction
#   - platform.claude.com/docs/build-with-claude/context-editing
#   - platform.claude.com/docs/build-with-claude/prompt-caching
# 
# Strategy:
#   1. Adaptive thinking (default): Fast, free, suitable for most compaction scenarios
#   2. Extended thinking (fallback): Only when adaptive is insufficient or task is complex
#   3. Thinking exclusion: Leverage automatic thinking block filtering in API
# 
# USAGE:
#   compactor = GPT5AdaptiveCompactor()
#   compacted = compactor.compact(
#       conversation=history,
#       token_budget=200_000,
#       strategy='adaptive',  # or 'extended'
#   )
#   metadata = compactor.get_compaction_metadata()
# """
# 
# import logging
# import json
# import os
# from pathlib import Path
# from typing import Optional, TypedDict
# from dataclasses import dataclass, asdict
# from datetime import datetime, timezone
# 
# log = logging.getLogger(__name__)
# 
# # Environment variable to control backend (default: copilot-sdk, which uses gpt-5-mini)
# BEAST_BACKEND = os.environ.get("BEAST_BACKEND", "copilot-sdk")
# 
# 
# class CompactionMetadata(TypedDict):
#     """Metadata about compaction operation."""
#     tokens_before: int
#     tokens_after: int
#     reduction_pct: float
#     strategy_used: str  # 'adaptive' | 'extended'
#     thinking_used: bool
#     thinking_tokens: int
#     model: str
#     timestamp: str
#     success: bool
# 
# 
# @dataclass
# class CompactionResult:
#     """Result of compaction operation."""
#     conversation: list[dict]
#     metadata: CompactionMetadata
#     is_sufficient: bool  # True if reduced below threshold
# 
# 
# class GPT5AdaptiveCompactor:
#     """
#     Model-based context compactor using gpt-5-mini with adaptive thinking.
#     
#     Integrates with AutoBeast's model selection (via Copilot SDK / worker.py).
#     Uses free gpt-5-mini tier; escalates to extended thinking only when needed.
#     """
# 
#     def __init__(
#         self,
#         model: str = "gpt-5-mini",
#         token_limit: int = 200_000,
#         compaction_threshold_pct: float = 85.0,
#         enable_extended_thinking: bool = True,
#     ):
#         """
#         Args:
#             model: Model to use (default: gpt-5-mini, free)
#             token_limit: Maximum context window
#             compaction_threshold_pct: Target threshold after compaction
#             enable_extended_thinking: Allow fallback to extended thinking if adaptive insufficient
#         """
#         self.model = model
#         self.token_limit = token_limit
#         self.threshold_pct = compaction_threshold_pct
#         self.enable_extended = enable_extended_thinking
#         self.backend = BEAST_BACKEND
#         
#         log.info(
#             f"[gpt5-compactor] initialized: model={model}, backend={self.backend}, "
#             f"token_limit={token_limit:,}, threshold={compaction_threshold_pct}%"
#         )
# 
#     def compact(
#         self,
#         conversation: list[dict],
#         token_budget: Optional[int] = None,
#         strategy: str = "adaptive",
#     ) -> CompactionResult:
#         """
#         Compact conversation using gpt-5-mini with adaptive thinking.
# 
#         Args:
#             conversation: Message history
#             token_budget: Optional override for token limit (uses self.token_limit if None)
#             strategy: 'adaptive' (default) | 'extended' (expensive fallback)
# 
#         Returns:
#             CompactionResult with compacted conversation + metadata
#         """
#         if not conversation:
#             return CompactionResult(
#                 conversation=[],
#                 metadata={
#                     "tokens_before": 0,
#                     "tokens_after": 0,
#                     "reduction_pct": 0.0,
#                     "strategy_used": strategy,
#                     "thinking_used": False,
#                     "thinking_tokens": 0,
#                     "model": self.model,
#                     "timestamp": datetime.now(timezone.utc).isoformat(),
#                     "success": True,
#                 },
#                 is_sufficient=True,
#             )
# 
#         limit = token_budget or self.token_limit
# 
#         # Build compaction prompt for the model
#         compaction_prompt = self._build_compaction_prompt(conversation, limit)
# 
#         # Call model with adaptive thinking (free) → fallback to extended if needed
#         if strategy == "adaptive":
#             result = self._call_with_adaptive_thinking(compaction_prompt, conversation, limit)
#         elif strategy == "extended" and self.enable_extended:
#             result = self._call_with_extended_thinking(compaction_prompt, conversation, limit)
#         else:
#             # Fallback to heuristic if model call unavailable
#             result = self._fallback_heuristic_compact(conversation, limit)
# 
#         log.info(
#             f"[gpt5-compactor] compact complete: {result.metadata['tokens_before']:,} → "
#             f"{result.metadata['tokens_after']:,} tokens "
#             f"({result.metadata['reduction_pct']:.1f}% reduction, "
#             f"strategy={result.metadata['strategy_used']}, "
#             f"thinking_tokens={result.metadata['thinking_tokens']}, "
#             f"success={result.metadata['success']})"
#         )
# 
#         return result
# 
#     def _build_compaction_prompt(self, conversation: list[dict], token_limit: int) -> str:
#         """Build a prompt instructing the model to compact the conversation."""
#         current_tokens = self._estimate_tokens(json.dumps(conversation))
#         target_tokens = int(token_limit * (self.threshold_pct / 100.0))
# 
#         prompt = f"""You are a context compaction specialist. Analyze this conversation and produce a compacted version.
# 
# Current token usage: ~{current_tokens:,} tokens
# Token limit: {token_limit:,} tokens
# Target usage: ~{target_tokens:,} tokens (below {self.threshold_pct}% threshold)
# 
# Task:
# 1. Identify key information that MUST be preserved (user intent, critical context, decisions)
# 2. Identify redundant or tool-output content that can be safely removed/summarized
# 3. Keep system messages and recent turns intact
# 4. Clean up verbose tool outputs, keeping only structured results
# 5. Provide a JSON array of compacted messages
# 
# Output format: valid JSON array only, no explanation.
# 
# Conversation to compact:
# {json.dumps(conversation, indent=2)[:5000]}...
# 
# Compacted conversation (valid JSON array):"""
#         return prompt
# 
#     def _estimate_tokens(self, text: str) -> int:
#         """Quick token estimation (1 token ≈ 3-4 chars for general text)."""
#         if not text:
#             return 0
#         return max(1, len(text) // 3)
# 
#     def _call_with_adaptive_thinking(
#         self, prompt: str, conversation: list[dict], token_limit: int
#     ) -> CompactionResult:
#         """
#         Call gpt-5-mini with adaptive thinking (fast, free).
#         
#         Adaptive thinking allows gpt-5-mini to reason through problems efficiently
#         without the cost of extended thinking. Perfect for routine compaction tasks.
#         """
#         try:
#             # Try to use worker.py's model invocation if available
#             from .worker import WorkerAgent
#             
#             agent = WorkerAgent("code_agent", complexity="simple")
#             
#             # Build a minimal task for the model
#             task = {
#                 "id": "compaction_adaptive",
#                 "agent": "code_agent",
#                 "type": "reasoning",
#                 "description": "Compact conversation using adaptive thinking",
#                 "request": prompt,
#                 "complexity": "simple",
#                 "model": self.model,
#                 "thinking": {"adaptive": True},  # Enable adaptive thinking
#             }
# 
#             log.info(f"[gpt5-compactor] calling {self.model} with adaptive thinking...")
#             result = agent.run(task, context={})
# 
#             if result.get("status") == "completed":
#                 compacted = self._parse_compaction_response(result.get("output", ""), conversation)
#                 tokens_after = self._estimate_tokens(json.dumps(compacted))
#                 tokens_before = self._estimate_tokens(json.dumps(conversation))
#                 reduction = ((tokens_before - tokens_after) / max(1, tokens_before)) * 100
# 
#                 return CompactionResult(
#                     conversation=compacted,
#                     metadata={
#                         "tokens_before": tokens_before,
#                         "tokens_after": tokens_after,
#                         "reduction_pct": reduction,
#                         "strategy_used": "adaptive",
#                         "thinking_used": True,
#                         "thinking_tokens": 0,  # Not tracked in response
#                         "model": self.model,
#                         "timestamp": datetime.now(timezone.utc).isoformat(),
#                         "success": True,
#                     },
#                     is_sufficient=(tokens_after / token_limit) < (self.threshold_pct / 100.0),
#                 )
#             else:
#                 log.warning(f"[gpt5-compactor] adaptive thinking failed: {result.get('output')}")
#                 return self._fallback_heuristic_compact(conversation, token_limit)
# 
#         except Exception as e:
#             log.warning(f"[gpt5-compactor] adaptive thinking call failed: {e}")
#             return self._fallback_heuristic_compact(conversation, token_limit)
# 
#     def _call_with_extended_thinking(
#         self, prompt: str, conversation: list[dict], token_limit: int
#     ) -> CompactionResult:
#         """
#         Call gpt-5-mini with extended thinking (slower, but thorough).
#         
#         Extended thinking tokens are automatically excluded from previous turns
#         when passed back as conversation history, so this is still token-efficient.
#         Only used when adaptive thinking is insufficient.
#         """
#         try:
#             from .worker import WorkerAgent
#             
#             agent = WorkerAgent("code_agent", complexity="medium")
#             
#             task = {
#                 "id": "compaction_extended",
#                 "agent": "code_agent",
#                 "type": "reasoning",
#                 "description": "Compact conversation using extended thinking",
#                 "request": prompt,
#                 "complexity": "medium",
#                 "model": self.model,
#                 "thinking": {
#                     "adaptive": False,
#                     "extended": True,  # Enable extended thinking (fallback)
#                     "max_thinking_tokens": 10_000,  # Limit to control cost
#                 },
#             }
# 
#             log.info(f"[gpt5-compactor] calling {self.model} with extended thinking (fallback)...")
#             result = agent.run(task, context={})
# 
#             if result.get("status") == "completed":
#                 compacted = self._parse_compaction_response(result.get("output", ""), conversation)
#                 tokens_after = self._estimate_tokens(json.dumps(compacted))
#                 tokens_before = self._estimate_tokens(json.dumps(conversation))
#                 reduction = ((tokens_before - tokens_after) / max(1, tokens_before)) * 100
# 
#                 return CompactionResult(
#                     conversation=compacted,
#                     metadata={
#                         "tokens_before": tokens_before,
#                         "tokens_after": tokens_after,
#                         "reduction_pct": reduction,
#                         "strategy_used": "extended",
#                         "thinking_used": True,
#                         "thinking_tokens": 10_000,  # Estimate
#                         "model": self.model,
#                         "timestamp": datetime.now(timezone.utc).isoformat(),
#                         "success": True,
#                     },
#                     is_sufficient=(tokens_after / token_limit) < (self.threshold_pct / 100.0),
#                 )
#             else:
#                 log.warning(f"[gpt5-compactor] extended thinking failed: {result.get('output')}")
#                 return self._fallback_heuristic_compact(conversation, token_limit)
# 
#         except Exception as e:
#             log.warning(f"[gpt5-compactor] extended thinking call failed: {e}")
#             return self._fallback_heuristic_compact(conversation, token_limit)
# 
#     def _parse_compaction_response(self, response_text: str, original: list[dict]) -> list[dict]:
#         """
#         Parse model's compaction response.
#         
#         Tries to extract JSON array; falls back to heuristic if parsing fails.
#         """
#         try:
#             # Try to find JSON array in response
#             response_text = response_text.strip()
#             if response_text.startswith("["):
#                 compacted = json.loads(response_text)
#                 if isinstance(compacted, list):
#                     log.info(f"[gpt5-compactor] parsed compacted response: {len(original)} → {len(compacted)} messages")
#                     return compacted
#         except json.JSONDecodeError:
#             pass
# 
#         log.warning("[gpt5-compactor] failed to parse model response; using heuristic fallback")
#         return self._fallback_heuristic_compact(original, self.token_limit).conversation
# 
#     def _fallback_heuristic_compact(
#         self, conversation: list[dict], token_limit: int
#     ) -> CompactionResult:
#         """
#         Fallback heuristic compaction (no model call).
#         
#         Used when model is unavailable or calls fail.
#         Implements the same strategies as context_compactor.py.
#         """
#         from .context_compactor import ContextCompactor
#         
#         heuristic = ContextCompactor(token_limit=token_limit)
#         compacted = heuristic.compact(conversation, strategy="combined", keep_last_n_turns=10)
#         
#         tokens_before = heuristic.estimate_tokens(conversation).tokens_used
#         tokens_after = heuristic.estimate_tokens(compacted).tokens_used
#         reduction = ((tokens_before - tokens_after) / max(1, tokens_before)) * 100
# 
#         return CompactionResult(
#             conversation=compacted,
#             metadata={
#                 "tokens_before": tokens_before,
#                 "tokens_after": tokens_after,
#                 "reduction_pct": reduction,
#                 "strategy_used": "heuristic_fallback",
#                 "thinking_used": False,
#                 "thinking_tokens": 0,
#                 "model": "heuristic",
#                 "timestamp": datetime.now(timezone.utc).isoformat(),
#                 "success": True,
#             },
#             is_sufficient=(tokens_after / token_limit) < (self.threshold_pct / 100.0),
#         )
# 
#     def get_compaction_metadata(self) -> dict:
#         """Return current compactor configuration as metadata."""
#         return {
#             "model": self.model,
#             "backend": self.backend,
#             "token_limit": self.token_limit,
#             "threshold_pct": self.threshold_pct,
#             "enable_extended_thinking": self.enable_extended,
#             "description": "GPT-5-Mini Adaptive Thinking Compactor (free tier)",
#         }
