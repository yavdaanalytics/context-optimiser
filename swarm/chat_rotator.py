"""ChatRotator coordinates session files and rotation logic.

When a session exceeds the token limit, this rotates it to a new session
while maintaining a linked chain.
"""
from pathlib import Path
from typing import Optional
import json
import uuid
from datetime import datetime

class ChatRotator:
    def __init__(self, session_dir: str = ".agent/data/state"):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.session_file = self.session_dir / "session.json"

    def find_session_file(self) -> Optional[Path]:
        if self.session_file.exists():
            return self.session_file
        return None

    def load_session(self, path: Path) -> Optional[dict]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def create_new_session(self, current_session: dict, compaction_stats: dict, origin_prompt: str) -> dict:
        new_id = str(uuid.uuid4())[:8]
        return {
            "session_id": new_id,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "phase": "initialized",
            "previous_session_id": current_session.get("session_id", "unknown"),
            "origin_prompt": origin_prompt,
            "subtask_graph": []
        }

    def write_new_session(self, new_session: dict) -> Path:
        new_path = self.session_dir / f"session_{new_session['session_id']}.json"
        new_path.write_text(json.dumps(new_session, indent=2), encoding="utf-8")
        return new_path

    def update_current_session_link(self, current_session_file: Path, next_session_id: str):
        try:
            data = json.loads(current_session_file.read_text(encoding="utf-8"))
            data["next_session_id"] = next_session_id
            current_session_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def create_chain_link(self, current_session: dict, new_session: dict, compaction_reduction_pct: float, origin_prompt: str) -> dict:
        return {
            "from_session": current_session.get("session_id", "unknown"),
            "to_session": new_session.get("session_id"),
            "reduction_pct": compaction_reduction_pct,
            "reason": "auto_rotation_threshold_exceeded"
        }

    def log_rotation(self, metadata: dict, new_session_file: Path):
        log_file = self.session_dir / "rotation.log"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(metadata) + "\n")

    def get_notification_message(self, metadata: dict) -> str:
        return f"Context rotated to new session {metadata['to_session']} (reduced context by {metadata.get('reduction_pct', 0):.1f}%)."
