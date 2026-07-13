"""
chat_rotator.py — File-Based Chat Session Rotation

When AutoBeast detects context limit exceeded (after post-task compaction),
rotates to a new chat session while preserving metadata and history links.

All operations are file-based (no API):
  1. Create new .agent/data/state/session_<timestamp>.json
  2. Write metadata: original_session, origin_prompt, backup_depth, compaction_stats
  3. Create chain link: session.json → session_N+1.json
  4. Notify user with: new session ID, link to previous, and next prompt instruction

USAGE:
  rotator = ChatRotator()
  if should_rotate:
      new_session = rotator.create_new_session(
          current_session_data,
          compaction_stats,
          origin_prompt,
      )
      metadata = rotator.create_chain_link(current_session_data, new_session)
      rotator.notify_user(metadata)
"""
import logging
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass, asdict
import uuid
log = logging.getLogger(__name__)

@dataclass
class ChainMetadata:
    """Metadata for session rotation chain."""
    current_session_id: str
    next_session_id: str
    previous_session_id: Optional[str]
    rotated_at: str
    reason: str
    compaction_reduction_pct: float
    tokens_at_rotation: int
    origin_prompt: str
    chain_depth: int

class ChatRotator:
    """File-based session rotation for context limit overflow."""
    SESSIONS_DIR = Path(__file__).resolve().parent.parent / 'data' / 'state'
    METADATA_SUFFIX = '.chain_metadata.json'
    ROTATION_LOG = Path(__file__).resolve().parent.parent / 'data' / 'state' / 'rotation_log.jsonl'

    def __init__(self):
        self.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    def create_new_session(self, current_session: dict, compaction_stats: dict, origin_prompt: str) -> dict:
        """
        Create a new session file with chain metadata.

        Args:
            current_session: Current session.json contents
            compaction_stats: Output from ContextCompactor.estimate_tokens()
            origin_prompt: User's prompt that triggered the need for rotation

        Returns:
            New session dict (ready to be written to filesystem)
        """
        current_session_id = current_session.get('session_id', 'unknown')
        new_session_id = f'session_{datetime.now(timezone.utc).isoformat().replace(':', '').replace('.', '_')}'
        chain_depth = current_session.get('chain_metadata', {}).get('chain_depth', 0) + 1
        previous_session_id = current_session.get('session_id')
        new_session = {'session_id': new_session_id, 'created_at': datetime.now(timezone.utc).isoformat(), 'rotated_from': current_session_id, 'tasks': [], 'chain_metadata': {'current': new_session_id, 'previous': previous_session_id, 'chain_depth': chain_depth, 'rotated_at': datetime.now(timezone.utc).isoformat(), 'reason': 'context_limit_exceeded', 'origin_prompt': origin_prompt[:200], 'compaction_reduction_pct': compaction_stats.get('pct_used', 0), 'tokens_at_rotation': compaction_stats.get('tokens_used', 0)}}
        log.info(f'[rotator] new session created: {new_session_id} (chain depth: {chain_depth})')
        return new_session

    def write_new_session(self, new_session: dict) -> Path:
        """
        Write new session to disk.

        Returns:
            Path to new session file
        """
        session_file = self.SESSIONS_DIR / f'{new_session['session_id']}.json'
        session_file.write_text(json.dumps(new_session, indent=2), encoding='utf-8')
        log.info(f'[rotator] session written: {session_file}')
        return session_file

    def update_current_session_link(self, current_session_file: Path, new_session_id: str):
        """
        Update current session.json to point to the new session.
        This acts as a breadcrumb for continuity.
        """
        try:
            current_session = json.loads(current_session_file.read_text(encoding='utf-8'))
            current_session['next_session'] = new_session_id
            current_session_file.write_text(json.dumps(current_session, indent=2), encoding='utf-8')
            log.info(f'[rotator] session link updated: {current_session_file} → {new_session_id}')
        except Exception as e:
            log.error(f'[rotator] failed to update session link: {e}')

    def create_chain_link(self, current_session: dict, new_session: dict, compaction_reduction_pct: float=0.0, origin_prompt: str='') -> ChainMetadata:
        """
        Create metadata linking old session to new.

        Returns:
            ChainMetadata object (for notification + logging)
        """
        metadata = ChainMetadata(current_session_id=current_session.get('session_id', 'unknown'), next_session_id=new_session.get('session_id', 'unknown'), previous_session_id=current_session.get('rotated_from'), rotated_at=datetime.now(timezone.utc).isoformat(), reason='context_limit_exceeded', compaction_reduction_pct=compaction_reduction_pct, tokens_at_rotation=current_session.get('chain_metadata', {}).get('tokens_at_rotation', 0), origin_prompt=origin_prompt[:100], chain_depth=new_session.get('chain_metadata', {}).get('chain_depth', 1))
        return metadata

    def log_rotation(self, metadata: ChainMetadata, output_file: Optional[Path]=None):
        """
        Append rotation event to rotation_log.jsonl for audit trail.

        Args:
            metadata: ChainMetadata from create_chain_link()
            output_file: Optional path to newly created session file
        """
        self.ROTATION_LOG.parent.mkdir(parents=True, exist_ok=True)
        entry = {**asdict(metadata), 'session_file': str(output_file) if output_file else None, 'timestamp': datetime.now(timezone.utc).isoformat()}
        with self.ROTATION_LOG.open('a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')
        log.info(f'[rotator] rotation logged: {self.ROTATION_LOG}')

    def get_notification_message(self, metadata: ChainMetadata) -> str:
        """
        Generate a brief user-facing notification message.

        Returns:
            Notification text with links and instructions
        """
        msg = f'📢 **AutoBeast Context Rotation**\n\nContext window was {metadata.tokens_at_rotation:,} tokens (at rotation threshold).\nCompaction reduced prompt by ~{metadata.compaction_reduction_pct:.1f}%.\n\n✨ **New chat created:** `{metadata.next_session_id}`\n🔗 **Previous session:** `{metadata.current_session_id}`\n📊 **Chain depth:** {metadata.chain_depth}\n\nYour new prompt will start in the fresh chat. Previous session history is preserved in state files for reference.\n'
        return msg

    @staticmethod
    def find_session_file(session_dir: Path=None) -> Optional[Path]:
        """
        Find the current session.json file.

        Returns:
            Path to session.json, or None if not found
        """
        if session_dir is None:
            session_dir = ChatRotator.SESSIONS_DIR
        session_file = session_dir / 'session.json'
        if session_file.exists():
            return session_file
        log.warning(f'[rotator] session.json not found in {session_dir}')
        return None

    @staticmethod
    def load_session(session_file: Optional[Path]=None) -> Optional[dict]:
        """Load current session data."""
        if session_file is None:
            session_file = ChatRotator.find_session_file()
        if not session_file or not session_file.exists():
            return None
        try:
            return json.loads(session_file.read_text(encoding='utf-8'))
        except Exception as e:
            log.error(f'[rotator] failed to load session: {e}')
            return None

    @staticmethod
    def get_chain_history(session_id: Optional[str]=None, session_dir: Path=None) -> list[str]:
        """
        Trace back through session chain to find all linked sessions.

        Returns:
            List of session IDs, newest first
        """
        if session_dir is None:
            session_dir = ChatRotator.SESSIONS_DIR
        if session_id is None:
            session = ChatRotator.load_session()
            session_id = session.get('session_id') if session else None
        if not session_id:
            return []
        chain = [session_id]
        current_id = session_id
        max_hops = 50
        while len(chain) < max_hops:
            session_file = session_dir / f'{current_id}.json'
            if not session_file.exists():
                break
            try:
                data = json.loads(session_file.read_text(encoding='utf-8'))
                prev_id = data.get('rotated_from') or data.get('chain_metadata', {}).get('previous')
                if not prev_id or prev_id in chain:
                    break
                chain.append(prev_id)
                current_id = prev_id
            except Exception:
                break
        return chain

@dataclass
class ChainMetadata__from_archived_5338:
    """Metadata for session rotation chain."""
    current_session_id: str
    next_session_id: str
    previous_session_id: Optional[str]
    rotated_at: str
    reason: str
    compaction_reduction_pct: float
    tokens_at_rotation: int
    origin_prompt: str
    chain_depth: int

class ChatRotator__from_archived_5338:
    """File-based session rotation for context limit overflow."""
    SESSIONS_DIR = Path(__file__).resolve().parent.parent / 'data' / 'state'
    METADATA_SUFFIX = '.chain_metadata.json'
    ROTATION_LOG = Path(__file__).resolve().parent.parent / 'data' / 'state' / 'rotation_log.jsonl'

    def __init__(self):
        self.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    def create_new_session(self, current_session: dict, compaction_stats: dict, origin_prompt: str) -> dict:
        """
        Create a new session file with chain metadata.

        Args:
            current_session: Current session.json contents
            compaction_stats: Output from ContextCompactor.estimate_tokens()
            origin_prompt: User's prompt that triggered the need for rotation

        Returns:
            New session dict (ready to be written to filesystem)
        """
        current_session_id = current_session.get('session_id', 'unknown')
        new_session_id = f'session_{datetime.now(timezone.utc).isoformat().replace(':', '').replace('.', '_')}'
        chain_depth = current_session.get('chain_metadata', {}).get('chain_depth', 0) + 1
        previous_session_id = current_session.get('session_id')
        new_session = {'session_id': new_session_id, 'created_at': datetime.now(timezone.utc).isoformat(), 'rotated_from': current_session_id, 'tasks': [], 'chain_metadata': {'current': new_session_id, 'previous': previous_session_id, 'chain_depth': chain_depth, 'rotated_at': datetime.now(timezone.utc).isoformat(), 'reason': 'context_limit_exceeded', 'origin_prompt': origin_prompt[:200], 'compaction_reduction_pct': compaction_stats.get('pct_used', 0), 'tokens_at_rotation': compaction_stats.get('tokens_used', 0)}}
        log.info(f'[rotator] new session created: {new_session_id} (chain depth: {chain_depth})')
        return new_session

    def write_new_session(self, new_session: dict) -> Path:
        """
        Write new session to disk.

        Returns:
            Path to new session file
        """
        session_file = self.SESSIONS_DIR / f'{new_session['session_id']}.json'
        session_file.write_text(json.dumps(new_session, indent=2), encoding='utf-8')
        log.info(f'[rotator] session written: {session_file}')
        return session_file

    def update_current_session_link(self, current_session_file: Path, new_session_id: str):
        """
        Update current session.json to point to the new session.
        This acts as a breadcrumb for continuity.
        """
        try:
            current_session = json.loads(current_session_file.read_text(encoding='utf-8'))
            current_session['next_session'] = new_session_id
            current_session_file.write_text(json.dumps(current_session, indent=2), encoding='utf-8')
            log.info(f'[rotator] session link updated: {current_session_file} → {new_session_id}')
        except Exception as e:
            log.error(f'[rotator] failed to update session link: {e}')

    def create_chain_link(self, current_session: dict, new_session: dict, compaction_reduction_pct: float=0.0, origin_prompt: str='') -> ChainMetadata:
        """
        Create metadata linking old session to new.

        Returns:
            ChainMetadata object (for notification + logging)
        """
        metadata = ChainMetadata(current_session_id=current_session.get('session_id', 'unknown'), next_session_id=new_session.get('session_id', 'unknown'), previous_session_id=current_session.get('rotated_from'), rotated_at=datetime.now(timezone.utc).isoformat(), reason='context_limit_exceeded', compaction_reduction_pct=compaction_reduction_pct, tokens_at_rotation=current_session.get('chain_metadata', {}).get('tokens_at_rotation', 0), origin_prompt=origin_prompt[:100], chain_depth=new_session.get('chain_metadata', {}).get('chain_depth', 1))
        return metadata

    def log_rotation(self, metadata: ChainMetadata, output_file: Optional[Path]=None):
        """
        Append rotation event to rotation_log.jsonl for audit trail.

        Args:
            metadata: ChainMetadata from create_chain_link()
            output_file: Optional path to newly created session file
        """
        self.ROTATION_LOG.parent.mkdir(parents=True, exist_ok=True)
        entry = {**asdict(metadata), 'session_file': str(output_file) if output_file else None, 'timestamp': datetime.now(timezone.utc).isoformat()}
        with self.ROTATION_LOG.open('a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')
        log.info(f'[rotator] rotation logged: {self.ROTATION_LOG}')

    def get_notification_message(self, metadata: ChainMetadata) -> str:
        """
        Generate a brief user-facing notification message.

        Returns:
            Notification text with links and instructions
        """
        msg = f'📢 **AutoBeast Context Rotation**\n\nContext window was {metadata.tokens_at_rotation:,} tokens (at rotation threshold).\nCompaction reduced prompt by ~{metadata.compaction_reduction_pct:.1f}%.\n\n✨ **New chat created:** `{metadata.next_session_id}`\n🔗 **Previous session:** `{metadata.current_session_id}`\n📊 **Chain depth:** {metadata.chain_depth}\n\nYour new prompt will start in the fresh chat. Previous session history is preserved in state files for reference.\n'
        return msg

    @staticmethod
    def find_session_file(session_dir: Path=None) -> Optional[Path]:
        """
        Find the current session.json file.

        Returns:
            Path to session.json, or None if not found
        """
        if session_dir is None:
            session_dir = ChatRotator.SESSIONS_DIR
        session_file = session_dir / 'session.json'
        if session_file.exists():
            return session_file
        log.warning(f'[rotator] session.json not found in {session_dir}')
        return None

    @staticmethod
    def load_session(session_file: Optional[Path]=None) -> Optional[dict]:
        """Load current session data."""
        if session_file is None:
            session_file = ChatRotator.find_session_file()
        if not session_file or not session_file.exists():
            return None
        try:
            return json.loads(session_file.read_text(encoding='utf-8'))
        except Exception as e:
            log.error(f'[rotator] failed to load session: {e}')
            return None

    @staticmethod
    def get_chain_history(session_id: Optional[str]=None, session_dir: Path=None) -> list[str]:
        """
        Trace back through session chain to find all linked sessions.

        Returns:
            List of session IDs, newest first
        """
        if session_dir is None:
            session_dir = ChatRotator.SESSIONS_DIR
        if session_id is None:
            session = ChatRotator.load_session()
            session_id = session.get('session_id') if session else None
        if not session_id:
            return []
        chain = [session_id]
        current_id = session_id
        max_hops = 50
        while len(chain) < max_hops:
            session_file = session_dir / f'{current_id}.json'
            if not session_file.exists():
                break
            try:
                data = json.loads(session_file.read_text(encoding='utf-8'))
                prev_id = data.get('rotated_from') or data.get('chain_metadata', {}).get('previous')
                if not prev_id or prev_id in chain:
                    break
                chain.append(prev_id)
                current_id = prev_id
            except Exception:
                break
        return chain

@dataclass
class ChainMetadata__from_archived_6299:
    """Metadata for session rotation chain."""
    current_session_id: str
    next_session_id: str
    previous_session_id: Optional[str]
    rotated_at: str
    reason: str
    compaction_reduction_pct: float
    tokens_at_rotation: int
    origin_prompt: str
    chain_depth: int

class ChatRotator__from_archived_6299:
    """File-based session rotation for context limit overflow."""
    SESSIONS_DIR = Path(__file__).resolve().parent.parent / 'data' / 'state'
    METADATA_SUFFIX = '.chain_metadata.json'
    ROTATION_LOG = Path(__file__).resolve().parent.parent / 'data' / 'state' / 'rotation_log.jsonl'

    def __init__(self):
        self.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    def create_new_session(self, current_session: dict, compaction_stats: dict, origin_prompt: str) -> dict:
        """
        Create a new session file with chain metadata.

        Args:
            current_session: Current session.json contents
            compaction_stats: Output from ContextCompactor.estimate_tokens()
            origin_prompt: User's prompt that triggered the need for rotation

        Returns:
            New session dict (ready to be written to filesystem)
        """
        current_session_id = current_session.get('session_id', 'unknown')
        new_session_id = f'session_{datetime.now(timezone.utc).isoformat().replace(':', '').replace('.', '_')}'
        chain_depth = current_session.get('chain_metadata', {}).get('chain_depth', 0) + 1
        previous_session_id = current_session.get('session_id')
        new_session = {'session_id': new_session_id, 'created_at': datetime.now(timezone.utc).isoformat(), 'rotated_from': current_session_id, 'tasks': [], 'chain_metadata': {'current': new_session_id, 'previous': previous_session_id, 'chain_depth': chain_depth, 'rotated_at': datetime.now(timezone.utc).isoformat(), 'reason': 'context_limit_exceeded', 'origin_prompt': origin_prompt[:200], 'compaction_reduction_pct': compaction_stats.get('pct_used', 0), 'tokens_at_rotation': compaction_stats.get('tokens_used', 0)}}
        log.info(f'[rotator] new session created: {new_session_id} (chain depth: {chain_depth})')
        return new_session

    def write_new_session(self, new_session: dict) -> Path:
        """
        Write new session to disk.

        Returns:
            Path to new session file
        """
        session_file = self.SESSIONS_DIR / f'{new_session['session_id']}.json'
        session_file.write_text(json.dumps(new_session, indent=2), encoding='utf-8')
        log.info(f'[rotator] session written: {session_file}')
        return session_file

    def update_current_session_link(self, current_session_file: Path, new_session_id: str):
        """
        Update current session.json to point to the new session.
        This acts as a breadcrumb for continuity.
        """
        try:
            current_session = json.loads(current_session_file.read_text(encoding='utf-8'))
            current_session['next_session'] = new_session_id
            current_session_file.write_text(json.dumps(current_session, indent=2), encoding='utf-8')
            log.info(f'[rotator] session link updated: {current_session_file} → {new_session_id}')
        except Exception as e:
            log.error(f'[rotator] failed to update session link: {e}')

    def create_chain_link(self, current_session: dict, new_session: dict, compaction_reduction_pct: float=0.0, origin_prompt: str='') -> ChainMetadata:
        """
        Create metadata linking old session to new.

        Returns:
            ChainMetadata object (for notification + logging)
        """
        metadata = ChainMetadata(current_session_id=current_session.get('session_id', 'unknown'), next_session_id=new_session.get('session_id', 'unknown'), previous_session_id=current_session.get('rotated_from'), rotated_at=datetime.now(timezone.utc).isoformat(), reason='context_limit_exceeded', compaction_reduction_pct=compaction_reduction_pct, tokens_at_rotation=current_session.get('chain_metadata', {}).get('tokens_at_rotation', 0), origin_prompt=origin_prompt[:100], chain_depth=new_session.get('chain_metadata', {}).get('chain_depth', 1))
        return metadata

    def log_rotation(self, metadata: ChainMetadata, output_file: Optional[Path]=None):
        """
        Append rotation event to rotation_log.jsonl for audit trail.

        Args:
            metadata: ChainMetadata from create_chain_link()
            output_file: Optional path to newly created session file
        """
        self.ROTATION_LOG.parent.mkdir(parents=True, exist_ok=True)
        entry = {**asdict(metadata), 'session_file': str(output_file) if output_file else None, 'timestamp': datetime.now(timezone.utc).isoformat()}
        with self.ROTATION_LOG.open('a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')
        log.info(f'[rotator] rotation logged: {self.ROTATION_LOG}')

    def get_notification_message(self, metadata: ChainMetadata) -> str:
        """
        Generate a brief user-facing notification message.

        Returns:
            Notification text with links and instructions
        """
        msg = f'📢 **AutoBeast Context Rotation**\n\nContext window was {metadata.tokens_at_rotation:,} tokens (at rotation threshold).\nCompaction reduced prompt by ~{metadata.compaction_reduction_pct:.1f}%.\n\n✨ **New chat created:** `{metadata.next_session_id}`\n🔗 **Previous session:** `{metadata.current_session_id}`\n📊 **Chain depth:** {metadata.chain_depth}\n\nYour new prompt will start in the fresh chat. Previous session history is preserved in state files for reference.\n'
        return msg

    @staticmethod
    def find_session_file(session_dir: Path=None) -> Optional[Path]:
        """
        Find the current session.json file.

        Returns:
            Path to session.json, or None if not found
        """
        if session_dir is None:
            session_dir = ChatRotator.SESSIONS_DIR
        session_file = session_dir / 'session.json'
        if session_file.exists():
            return session_file
        log.warning(f'[rotator] session.json not found in {session_dir}')
        return None

    @staticmethod
    def load_session(session_file: Optional[Path]=None) -> Optional[dict]:
        """Load current session data."""
        if session_file is None:
            session_file = ChatRotator.find_session_file()
        if not session_file or not session_file.exists():
            return None
        try:
            return json.loads(session_file.read_text(encoding='utf-8'))
        except Exception as e:
            log.error(f'[rotator] failed to load session: {e}')
            return None

    @staticmethod
    def get_chain_history(session_id: Optional[str]=None, session_dir: Path=None) -> list[str]:
        """
        Trace back through session chain to find all linked sessions.

        Returns:
            List of session IDs, newest first
        """
        if session_dir is None:
            session_dir = ChatRotator.SESSIONS_DIR
        if session_id is None:
            session = ChatRotator.load_session()
            session_id = session.get('session_id') if session else None
        if not session_id:
            return []
        chain = [session_id]
        current_id = session_id
        max_hops = 50
        while len(chain) < max_hops:
            session_file = session_dir / f'{current_id}.json'
            if not session_file.exists():
                break
            try:
                data = json.loads(session_file.read_text(encoding='utf-8'))
                prev_id = data.get('rotated_from') or data.get('chain_metadata', {}).get('previous')
                if not prev_id or prev_id in chain:
                    break
                chain.append(prev_id)
                current_id = prev_id
            except Exception:
                break
        return chain

# === MERGED FROM C:\Users\AmitMohanty\.vscode\brain\.agent\tmp\moved_duplicates\20260328T133840Z\chat_rotator.py ===
# """
# chat_rotator.py — File-Based Chat Session Rotation
# 
# When AutoBeast detects context limit exceeded (after post-task compaction),
# rotates to a new chat session while preserving metadata and history links.
# 
# All operations are file-based (no API):
#   1. Create new .agent/data/state/session_<timestamp>.json
#   2. Write metadata: original_session, origin_prompt, backup_depth, compaction_stats
#   3. Create chain link: session.json → session_N+1.json
#   4. Notify user with: new session ID, link to previous, and next prompt instruction
# 
# USAGE:
#   rotator = ChatRotator()
#   if should_rotate:
#       new_session = rotator.create_new_session(
#           current_session_data,
#           compaction_stats,
#           origin_prompt,
#       )
#       metadata = rotator.create_chain_link(current_session_data, new_session)
#       rotator.notify_user(metadata)
# """
# 
# import logging
# import json
# from pathlib import Path
# from datetime import datetime, timezone
# from typing import Optional
# from dataclasses import dataclass, asdict
# import uuid
# 
# log = logging.getLogger(__name__)
# 
# 
# @dataclass
# class ChainMetadata:
#     """Metadata for session rotation chain."""
#     current_session_id: str
#     next_session_id: str
#     previous_session_id: Optional[str]
#     rotated_at: str  # ISO8601
#     reason: str  # e.g., "context_limit_exceeded"
#     compaction_reduction_pct: float  # Tokens reduced by compaction
#     tokens_at_rotation: int
#     origin_prompt: str  # User's prompt that triggered rotation
#     chain_depth: int  # 1 for first, 2 for second, etc.
# 
# 
# class ChatRotator:
#     """File-based session rotation for context limit overflow."""
# 
#     SESSIONS_DIR = Path(__file__).resolve().parent.parent / "data" / "state"
#     METADATA_SUFFIX = ".chain_metadata.json"
#     ROTATION_LOG = Path(__file__).resolve().parent.parent / "data" / "state" / "rotation_log.jsonl"
# 
#     def __init__(self):
#         self.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
# 
#     def create_new_session(
#         self,
#         current_session: dict,
#         compaction_stats: dict,
#         origin_prompt: str,
#     ) -> dict:
#         """
#         Create a new session file with chain metadata.
# 
#         Args:
#             current_session: Current session.json contents
#             compaction_stats: Output from ContextCompactor.estimate_tokens()
#             origin_prompt: User's prompt that triggered the need for rotation
# 
#         Returns:
#             New session dict (ready to be written to filesystem)
#         """
#         current_session_id = current_session.get("session_id", "unknown")
#         new_session_id = f"session_{datetime.now(timezone.utc).isoformat().replace(':', '').replace('.', '_')}"
# 
#         # Calculate chain depth
#         chain_depth = current_session.get("chain_metadata", {}).get("chain_depth", 0) + 1
#         previous_session_id = current_session.get("session_id")
# 
#         # New session skeleton
#         new_session = {
#             "session_id": new_session_id,
#             "created_at": datetime.now(timezone.utc).isoformat(),
#             "rotated_from": current_session_id,
#             "tasks": [],  # Fresh task list for new session
#             "chain_metadata": {
#                 "current": new_session_id,
#                 "previous": previous_session_id,
#                 "chain_depth": chain_depth,
#                 "rotated_at": datetime.now(timezone.utc).isoformat(),
#                 "reason": "context_limit_exceeded",
#                 "origin_prompt": origin_prompt[:200],  # Truncate for safety
#                 "compaction_reduction_pct": compaction_stats.get("pct_used", 0),
#                 "tokens_at_rotation": compaction_stats.get("tokens_used", 0),
#             },
#         }
# 
#         log.info(f"[rotator] new session created: {new_session_id} (chain depth: {chain_depth})")
#         return new_session
# 
#     def write_new_session(self, new_session: dict) -> Path:
#         """
#         Write new session to disk.
# 
#         Returns:
#             Path to new session file
#         """
#         session_file = self.SESSIONS_DIR / f"{new_session['session_id']}.json"
#         session_file.write_text(json.dumps(new_session, indent=2), encoding="utf-8")
#         log.info(f"[rotator] session written: {session_file}")
#         return session_file
# 
#     def update_current_session_link(self, current_session_file: Path, new_session_id: str):
#         """
#         Update current session.json to point to the new session.
#         This acts as a breadcrumb for continuity.
#         """
#         try:
#             current_session = json.loads(current_session_file.read_text(encoding="utf-8"))
#             current_session["next_session"] = new_session_id
#             current_session_file.write_text(json.dumps(current_session, indent=2), encoding="utf-8")
#             log.info(f"[rotator] session link updated: {current_session_file} → {new_session_id}")
#         except Exception as e:
#             log.error(f"[rotator] failed to update session link: {e}")
# 
#     def create_chain_link(
#         self,
#         current_session: dict,
#         new_session: dict,
#         compaction_reduction_pct: float = 0.0,
#         origin_prompt: str = "",
#     ) -> ChainMetadata:
#         """
#         Create metadata linking old session to new.
# 
#         Returns:
#             ChainMetadata object (for notification + logging)
#         """
#         metadata = ChainMetadata(
#             current_session_id=current_session.get("session_id", "unknown"),
#             next_session_id=new_session.get("session_id", "unknown"),
#             previous_session_id=current_session.get("rotated_from"),
#             rotated_at=datetime.now(timezone.utc).isoformat(),
#             reason="context_limit_exceeded",
#             compaction_reduction_pct=compaction_reduction_pct,
#             tokens_at_rotation=current_session.get("chain_metadata", {}).get("tokens_at_rotation", 0),
#             origin_prompt=origin_prompt[:100],
#             chain_depth=new_session.get("chain_metadata", {}).get("chain_depth", 1),
#         )
#         return metadata
# 
#     def log_rotation(self, metadata: ChainMetadata, output_file: Optional[Path] = None):
#         """
#         Append rotation event to rotation_log.jsonl for audit trail.
# 
#         Args:
#             metadata: ChainMetadata from create_chain_link()
#             output_file: Optional path to newly created session file
#         """
#         self.ROTATION_LOG.parent.mkdir(parents=True, exist_ok=True)
# 
#         entry = {
#             **asdict(metadata),
#             "session_file": str(output_file) if output_file else None,
#             "timestamp": datetime.now(timezone.utc).isoformat(),
#         }
# 
#         with self.ROTATION_LOG.open("a", encoding="utf-8") as f:
#             f.write(json.dumps(entry) + "\n")
# 
#         log.info(f"[rotator] rotation logged: {self.ROTATION_LOG}")
# 
#     def get_notification_message(self, metadata: ChainMetadata) -> str:
#         """
#         Generate a brief user-facing notification message.
# 
#         Returns:
#             Notification text with links and instructions
#         """
#         msg = (
#             f"📢 **AutoBeast Context Rotation**\n\n"
#             f"Context window was {metadata.tokens_at_rotation:,} tokens (at rotation threshold).\n"
#             f"Compaction reduced prompt by ~{metadata.compaction_reduction_pct:.1f}%.\n\n"
#             f"✨ **New chat created:** `{metadata.next_session_id}`\n"
#             f"🔗 **Previous session:** `{metadata.current_session_id}`\n"
#             f"📊 **Chain depth:** {metadata.chain_depth}\n\n"
#             f"Your new prompt will start in the fresh chat. "
#             f"Previous session history is preserved in state files for reference.\n"
#         )
#         return msg
# 
#     @staticmethod
#     def find_session_file(session_dir: Path = None) -> Optional[Path]:
#         """
#         Find the current session.json file.
# 
#         Returns:
#             Path to session.json, or None if not found
#         """
#         if session_dir is None:
#             session_dir = ChatRotator.SESSIONS_DIR
# 
#         session_file = session_dir / "session.json"
#         if session_file.exists():
#             return session_file
# 
#         log.warning(f"[rotator] session.json not found in {session_dir}")
#         return None
# 
#     @staticmethod
#     def load_session(session_file: Optional[Path] = None) -> Optional[dict]:
#         """Load current session data."""
#         if session_file is None:
#             session_file = ChatRotator.find_session_file()
# 
#         if not session_file or not session_file.exists():
#             return None
# 
#         try:
#             return json.loads(session_file.read_text(encoding="utf-8"))
#         except Exception as e:
#             log.error(f"[rotator] failed to load session: {e}")
#             return None
# 
#     @staticmethod
#     def get_chain_history(session_id: Optional[str] = None, session_dir: Path = None) -> list[str]:
#         """
#         Trace back through session chain to find all linked sessions.
# 
#         Returns:
#             List of session IDs, newest first
#         """
#         if session_dir is None:
#             session_dir = ChatRotator.SESSIONS_DIR
# 
#         if session_id is None:
#             session = ChatRotator.load_session()
#             session_id = session.get("session_id") if session else None
# 
#         if not session_id:
#             return []
# 
#         chain = [session_id]
#         current_id = session_id
# 
#         # Walk backwards through previous_session pointers
#         max_hops = 50  # Prevent infinite loops
#         while len(chain) < max_hops:
#             session_file = session_dir / f"{current_id}.json"
#             if not session_file.exists():
#                 break
# 
#             try:
#                 data = json.loads(session_file.read_text(encoding="utf-8"))
#                 prev_id = data.get("rotated_from") or data.get("chain_metadata", {}).get("previous")
#                 if not prev_id or prev_id in chain:
#                     break
#                 chain.append(prev_id)
#                 current_id = prev_id
#             except Exception:
#                 break
# 
#         return chain


# === MERGED FROM C:\Users\AmitMohanty\.vscode\brain\.agent\tmp\moved_duplicates\20260328T133840Z\chat_rotator.py.1 ===
# """
# chat_rotator.py — File-Based Chat Session Rotation
# 
# When AutoBeast detects context limit exceeded (after post-task compaction),
# rotates to a new chat session while preserving metadata and history links.
# 
# All operations are file-based (no API):
#   1. Create new .agent/data/state/session_<timestamp>.json
#   2. Write metadata: original_session, origin_prompt, backup_depth, compaction_stats
#   3. Create chain link: session.json → session_N+1.json
#   4. Notify user with: new session ID, link to previous, and next prompt instruction
# 
# USAGE:
#   rotator = ChatRotator()
#   if should_rotate:
#       new_session = rotator.create_new_session(
#           current_session_data,
#           compaction_stats,
#           origin_prompt,
#       )
#       metadata = rotator.create_chain_link(current_session_data, new_session)
#       rotator.notify_user(metadata)
# """
# 
# import logging
# import json
# from pathlib import Path
# from datetime import datetime, timezone
# from typing import Optional
# from dataclasses import dataclass, asdict
# import uuid
# 
# log = logging.getLogger(__name__)
# 
# 
# @dataclass
# class ChainMetadata:
#     """Metadata for session rotation chain."""
#     current_session_id: str
#     next_session_id: str
#     previous_session_id: Optional[str]
#     rotated_at: str  # ISO8601
#     reason: str  # e.g., "context_limit_exceeded"
#     compaction_reduction_pct: float  # Tokens reduced by compaction
#     tokens_at_rotation: int
#     origin_prompt: str  # User's prompt that triggered rotation
#     chain_depth: int  # 1 for first, 2 for second, etc.
# 
# 
# class ChatRotator:
#     """File-based session rotation for context limit overflow."""
# 
#     SESSIONS_DIR = Path(__file__).resolve().parent.parent / "data" / "state"
#     METADATA_SUFFIX = ".chain_metadata.json"
#     ROTATION_LOG = Path(__file__).resolve().parent.parent / "data" / "state" / "rotation_log.jsonl"
# 
#     def __init__(self):
#         self.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
# 
#     def create_new_session(
#         self,
#         current_session: dict,
#         compaction_stats: dict,
#         origin_prompt: str,
#     ) -> dict:
#         """
#         Create a new session file with chain metadata.
# 
#         Args:
#             current_session: Current session.json contents
#             compaction_stats: Output from ContextCompactor.estimate_tokens()
#             origin_prompt: User's prompt that triggered the need for rotation
# 
#         Returns:
#             New session dict (ready to be written to filesystem)
#         """
#         current_session_id = current_session.get("session_id", "unknown")
#         new_session_id = f"session_{datetime.now(timezone.utc).isoformat().replace(':', '').replace('.', '_')}"
# 
#         # Calculate chain depth
#         chain_depth = current_session.get("chain_metadata", {}).get("chain_depth", 0) + 1
#         previous_session_id = current_session.get("session_id")
# 
#         # New session skeleton
#         new_session = {
#             "session_id": new_session_id,
#             "created_at": datetime.now(timezone.utc).isoformat(),
#             "rotated_from": current_session_id,
#             "tasks": [],  # Fresh task list for new session
#             "chain_metadata": {
#                 "current": new_session_id,
#                 "previous": previous_session_id,
#                 "chain_depth": chain_depth,
#                 "rotated_at": datetime.now(timezone.utc).isoformat(),
#                 "reason": "context_limit_exceeded",
#                 "origin_prompt": origin_prompt[:200],  # Truncate for safety
#                 "compaction_reduction_pct": compaction_stats.get("pct_used", 0),
#                 "tokens_at_rotation": compaction_stats.get("tokens_used", 0),
#             },
#         }
# 
#         log.info(f"[rotator] new session created: {new_session_id} (chain depth: {chain_depth})")
#         return new_session
# 
#     def write_new_session(self, new_session: dict) -> Path:
#         """
#         Write new session to disk.
# 
#         Returns:
#             Path to new session file
#         """
#         session_file = self.SESSIONS_DIR / f"{new_session['session_id']}.json"
#         session_file.write_text(json.dumps(new_session, indent=2), encoding="utf-8")
#         log.info(f"[rotator] session written: {session_file}")
#         return session_file
# 
#     def update_current_session_link(self, current_session_file: Path, new_session_id: str):
#         """
#         Update current session.json to point to the new session.
#         This acts as a breadcrumb for continuity.
#         """
#         try:
#             current_session = json.loads(current_session_file.read_text(encoding="utf-8"))
#             current_session["next_session"] = new_session_id
#             current_session_file.write_text(json.dumps(current_session, indent=2), encoding="utf-8")
#             log.info(f"[rotator] session link updated: {current_session_file} → {new_session_id}")
#         except Exception as e:
#             log.error(f"[rotator] failed to update session link: {e}")
# 
#     def create_chain_link(
#         self,
#         current_session: dict,
#         new_session: dict,
#         compaction_reduction_pct: float = 0.0,
#         origin_prompt: str = "",
#     ) -> ChainMetadata:
#         """
#         Create metadata linking old session to new.
# 
#         Returns:
#             ChainMetadata object (for notification + logging)
#         """
#         metadata = ChainMetadata(
#             current_session_id=current_session.get("session_id", "unknown"),
#             next_session_id=new_session.get("session_id", "unknown"),
#             previous_session_id=current_session.get("rotated_from"),
#             rotated_at=datetime.now(timezone.utc).isoformat(),
#             reason="context_limit_exceeded",
#             compaction_reduction_pct=compaction_reduction_pct,
#             tokens_at_rotation=current_session.get("chain_metadata", {}).get("tokens_at_rotation", 0),
#             origin_prompt=origin_prompt[:100],
#             chain_depth=new_session.get("chain_metadata", {}).get("chain_depth", 1),
#         )
#         return metadata
# 
#     def log_rotation(self, metadata: ChainMetadata, output_file: Optional[Path] = None):
#         """
#         Append rotation event to rotation_log.jsonl for audit trail.
# 
#         Args:
#             metadata: ChainMetadata from create_chain_link()
#             output_file: Optional path to newly created session file
#         """
#         self.ROTATION_LOG.parent.mkdir(parents=True, exist_ok=True)
# 
#         entry = {
#             **asdict(metadata),
#             "session_file": str(output_file) if output_file else None,
#             "timestamp": datetime.now(timezone.utc).isoformat(),
#         }
# 
#         with self.ROTATION_LOG.open("a", encoding="utf-8") as f:
#             f.write(json.dumps(entry) + "\n")
# 
#         log.info(f"[rotator] rotation logged: {self.ROTATION_LOG}")
# 
#     def get_notification_message(self, metadata: ChainMetadata) -> str:
#         """
#         Generate a brief user-facing notification message.
# 
#         Returns:
#             Notification text with links and instructions
#         """
#         msg = (
#             f"📢 **AutoBeast Context Rotation**\n\n"
#             f"Context window was {metadata.tokens_at_rotation:,} tokens (at rotation threshold).\n"
#             f"Compaction reduced prompt by ~{metadata.compaction_reduction_pct:.1f}%.\n\n"
#             f"✨ **New chat created:** `{metadata.next_session_id}`\n"
#             f"🔗 **Previous session:** `{metadata.current_session_id}`\n"
#             f"📊 **Chain depth:** {metadata.chain_depth}\n\n"
#             f"Your new prompt will start in the fresh chat. "
#             f"Previous session history is preserved in state files for reference.\n"
#         )
#         return msg
# 
#     @staticmethod
#     def find_session_file(session_dir: Path = None) -> Optional[Path]:
#         """
#         Find the current session.json file.
# 
#         Returns:
#             Path to session.json, or None if not found
#         """
#         if session_dir is None:
#             session_dir = ChatRotator.SESSIONS_DIR
# 
#         session_file = session_dir / "session.json"
#         if session_file.exists():
#             return session_file
# 
#         log.warning(f"[rotator] session.json not found in {session_dir}")
#         return None
# 
#     @staticmethod
#     def load_session(session_file: Optional[Path] = None) -> Optional[dict]:
#         """Load current session data."""
#         if session_file is None:
#             session_file = ChatRotator.find_session_file()
# 
#         if not session_file or not session_file.exists():
#             return None
# 
#         try:
#             return json.loads(session_file.read_text(encoding="utf-8"))
#         except Exception as e:
#             log.error(f"[rotator] failed to load session: {e}")
#             return None
# 
#     @staticmethod
#     def get_chain_history(session_id: Optional[str] = None, session_dir: Path = None) -> list[str]:
#         """
#         Trace back through session chain to find all linked sessions.
# 
#         Returns:
#             List of session IDs, newest first
#         """
#         if session_dir is None:
#             session_dir = ChatRotator.SESSIONS_DIR
# 
#         if session_id is None:
#             session = ChatRotator.load_session()
#             session_id = session.get("session_id") if session else None
# 
#         if not session_id:
#             return []
# 
#         chain = [session_id]
#         current_id = session_id
# 
#         # Walk backwards through previous_session pointers
#         max_hops = 50  # Prevent infinite loops
#         while len(chain) < max_hops:
#             session_file = session_dir / f"{current_id}.json"
#             if not session_file.exists():
#                 break
# 
#             try:
#                 data = json.loads(session_file.read_text(encoding="utf-8"))
#                 prev_id = data.get("rotated_from") or data.get("chain_metadata", {}).get("previous")
#                 if not prev_id or prev_id in chain:
#                     break
#                 chain.append(prev_id)
#                 current_id = prev_id
#             except Exception:
#                 break
# 
#         return chain
