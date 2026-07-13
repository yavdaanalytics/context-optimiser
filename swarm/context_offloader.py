"""swarm/context_offloader.py

Utilities to offload older conversation messages to disk (scratchpad) while
keeping the last N messages in active context. Offloaded items are sanitized to
remove large code blocks, raw tool outputs, logs and file contents but retain
actionable lines and tags so the user doesn't feel the context is gone.

This module intentionally REDACTS noisy content instead of summarizing.
"""
from __future__ import annotations
import re
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from .vector_store import VectorMemoryStore


REDACT_CODE_PLACEHOLDER = "[[REDACTED CODE BLOCK: {n_lines} lines]]"
REDACT_LOG_PLACEHOLDER = "[[REDACTED LOGS]]"


def _sanitize_content(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""

    # Redact fenced code blocks ```...``` (keep an indicator with line count)
    def _code_repl(m):
        block = m.group(0)
        n_lines = block.count("\n")
        return REDACT_CODE_PLACEHOLDER.format(n_lines=n_lines)

    text = re.sub(r"```[\s\S]*?```", _code_repl, text)

    # Redact very large inline blocks (>200 lines) by keeping head/tail
    lines = text.splitlines()
    if len(lines) > 200:
        head = "\n".join(lines[:20])
        tail = "\n".join(lines[-20:])
        return head + "\n\n[[REDACTED MIDDLE OF LARGE BLOCK]]\n\n" + tail

    # Remove obvious stack traces or large log sections
    text = re.sub(r"Traceback \(most recent call last\):[\s\S]*?(?=\n\S|$)", REDACT_LOG_PLACEHOLDER, text)
    text = re.sub(r"(?m)^\s*(ERROR|Error|Exception|CRITICAL):.*$", REDACT_LOG_PLACEHOLDER, text)

    # Redact very long lines that look like file contents
    cleaned_lines = []
    for ln in text.splitlines():
        if len(ln) > 2000:
            cleaned_lines.append("[[REDACTED LONG LINE]]")
        else:
            cleaned_lines.append(ln)

    return "\n".join(cleaned_lines)


def _extract_tags(text: str) -> Dict[str, List[str]]:
    tags = {"decisions": [], "actions": [], "questions": []}
    if not text:
        return tags

    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if re.search(r"\b(TODO|Action:|Action item|Next steps|Next:|To do:)\b", s, re.I):
            tags["actions"].append(s[:300])
        if re.search(r"\b(Decision:|Decided:|Agreed:|Approved|we will|we'll|I'll |I will )\b", s, re.I):
            tags["decisions"].append(s[:300])
        if s.endswith("?") or re.match(r"^(Q\d+\.|Q:|Question:)", s):
            tags["questions"].append(s[:300])

    return tags


def offload_conversation(conversation: List[Dict[str, Any]], keep_last_n: int = 4, base_dir: str = ".agent/data/offloaded_conversations") -> List[str]:
    """Offload older messages from `conversation` while keeping the last
    `keep_last_n` messages in active context.

    The function writes one JSON file per offloaded message into `base_dir` and
    returns the list of file paths written. It REDACTS noisy content (code,
    logs, file dumps) but preserves short lines that look like actions,
    decisions, or questions so the conversation remains reconstructable.
    """
    p = Path(base_dir)
    p.mkdir(parents=True, exist_ok=True)

    if not isinstance(conversation, list):
        return []

    if len(conversation) <= keep_last_n:
        return []

    to_offload = conversation[:-keep_last_n]
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    written = []

    for idx, msg in enumerate(to_offload):
        role = None
        content = ""
        try:
            if isinstance(msg, dict):
                role = msg.get("role") or msg.get("r")
                content = msg.get("content") or msg.get("text") or ""
            else:
                content = str(msg)
        except Exception:
            content = str(msg)

        sanitized = _sanitize_content(content)
        tags = _extract_tags(content)

        record = {
            "index": idx,
            "role": role,
            "sanitized": sanitized,
            "original_length": len(content),
            "tags": tags,
        }

        fname = p / f"offload_{ts}_{idx}.json"
        try:
            with fname.open("w", encoding="utf-8") as fh:
                json.dump(record, fh, indent=2, ensure_ascii=False)
            written.append(str(fname))
        except Exception:
            # Non-fatal: continue trying to write other messages
            continue

    return written


def write_markdown_scratchpad(offload_paths: List[str], scratch_dir: str = ".agent/data/scratchpads") -> List[str]:
    """Create human-friendly markdown scratchpad files from offloaded JSON files.

    Returns list of written markdown file paths.
    """
    out = []
    base = Path(scratch_dir)
    base.mkdir(parents=True, exist_ok=True)

    for p in offload_paths:
        try:
            j = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:
            continue

        idx = j.get("index")
        role = j.get("role") or "unknown"
        tags = j.get("tags") or {}
        sanitized = j.get("sanitized") or ""

        md_lines = []
        md_lines.append(f"---\nrole: {role}\nindex: {idx}\nsource: {Path(p).name}\n---\n\n")
        if tags.get("decisions"):
            md_lines.append("## Decisions\n\n")
            for d in tags["decisions"]:
                md_lines.append(f"- {d}\n")
            md_lines.append("\n")
        if tags.get("actions"):
            md_lines.append("## Actions\n\n")
            for a in tags["actions"]:
                md_lines.append(f"- {a}\n")
            md_lines.append("\n")
        if tags.get("questions"):
            md_lines.append("## Questions\n\n")
            for q in tags["questions"]:
                md_lines.append(f"- {q}\n")
            md_lines.append("\n")

        md_lines.append("## Content (sanitized)\n\n")
        md_lines.append(sanitized + "\n")

        fname = base / f"offload_{Path(p).stem}.md"
        try:
            fname.write_text(''.join(md_lines), encoding="utf-8")
            out.append(str(fname))
        except Exception:
            continue

    return out


def distill_offloads(offload_paths: List[str], max_chars: int = 1200) -> str:
    """Produce a compact heuristic distillation from offloaded JSON files.

    This extracts decisions/actions/questions and head/tail excerpts to form a
    short summary usable for re-injection into active context without heavy LLM calls.
    """
    parts = []
    for p in offload_paths:
        try:
            j = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:
            continue
        role = j.get("role") or "unknown"
        tags = j.get("tags") or {}
        sanitized = j.get("sanitized") or ""

        header = f"[{role}] {Path(p).name} — {j.get('original_length',0)} chars"
        parts.append(header)
        # decisions/actions/questions prioritized
        for k in ("decisions", "actions", "questions"):
            items = tags.get(k) or []
            for it in items[:3]:
                parts.append(f"- {k[:-1].capitalize()}: {it}")

        # head/tail excerpt
        lines = sanitized.splitlines()
        if len(lines) > 40:
            excerpt = '\n'.join(lines[:8] + ['[[...]]'] + lines[-8:])
        else:
            excerpt = '\n'.join(lines[:40])
        parts.append(excerpt)

        if sum(len(p) for p in parts) > max_chars:
            break

    summary = '\n\n'.join(parts)
    if len(summary) > max_chars:
        summary = summary[:max_chars] + "\n..."
    return summary


def ingest_offloads_to_index(offload_paths: List[str], cw=None, mm=None):
    """Attempt to index offloaded content into the provided context window (cw)
    or memory manager (mm). This is best-effort and fail-soft.
    """
    # Try MM first
    if mm:
        try:
            if hasattr(mm, "record_offload"):
                for p in offload_paths:
                    try:
                        mm.record_offload(Path(p).read_text(encoding="utf-8"))
                    except Exception:
                        pass
                return True
            if hasattr(mm, "record"):
                for p in offload_paths:
                    try:
                        mm.record({"source": str(p), "content": Path(p).read_text(encoding="utf-8")})
                    except Exception:
                        pass
                return True
        except Exception:
            pass

    # Fallback: index into context window if available
    if cw:
        try:
            # Try a few likely method names
            for p in offload_paths:
                txt = Path(p).read_text(encoding="utf-8")
                try:
                    if hasattr(cw, "index_document"):
                        cw.index_document(str(p), txt)
                    elif hasattr(cw, "index_task_output"):
                        cw.index_task_output({"id": Path(p).stem}, txt)
                    elif hasattr(cw, "index"):
                        cw.index(txt)
                    else:
                        # As a last resort, try to append to an internal file
                        base = Path(__file__).resolve().parent.parent / ".agent" / "data" / "context_index"
                        base.mkdir(parents=True, exist_ok=True)
                        (base / f"{Path(p).stem}.txt").write_text(txt, encoding="utf-8")
                except Exception:
                    continue
            return True
        except Exception:
            pass

    return False


# VectorMemoryStore now imported from .vector_store


def compact_conversation_with_pointers(conversation: List[Dict[str, Any]], keep_last_n: int = 4, base_dir: str = ".agent/data/offloaded_conversations") -> List[Dict[str, Any]]:
    """
    Implements Intent Preservation & Semantic Pointers with Vector DB support:
    1. Preserves `role: user` messages that contain clear instructions.
    2. Offloads older completed tasks to ChromaDB (if available) or JSON disk files.
    3. Replaces offloaded content with a semantic pointer in the context array.
    """
    p = Path(base_dir)
    p.mkdir(parents=True, exist_ok=True)
    
    vdb = VectorMemoryStore()

    if not isinstance(conversation, list):
        return []

    if len(conversation) <= keep_last_n:
        return conversation

    to_offload = conversation[:-keep_last_n]
    keep_active = conversation[-keep_last_n:]
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    
    compacted = []

    for idx, msg in enumerate(to_offload):
        role = msg.get("role") or msg.get("r")
        content = str(msg.get("content") or msg.get("text") or "")
        
        # 1. Intent Preservation: Do not offload user prompts that have constraints
        if role == "user":
            # If the prompt is short or contains constraint keywords, keep it fully intact
            if len(content) < 500 or re.search(r"\b(do not|always|never|must|should|only)\b", content, re.I):
                compacted.append(msg)
                continue

        # 2. Extract tags for the semantic pointer
        tags = _extract_tags(content)
        sanitized = _sanitize_content(content)
        
        doc_id = f"task_{ts}_{idx}"
        
        metadata = {
            "role": role or "unknown",
            "original_length": len(content),
            "actions": " | ".join(tags.get("actions", [])[:3]),
            "decisions": " | ".join(tags.get("decisions", [])[:3])
        }
        
        # 2b. Write to Vector DB if available, else fallback to flat JSON
        fname = p / f"offload_{doc_id}.json"
        stored_in_vdb = vdb.store(doc_id, sanitized, metadata, collection_name="context_offloads")
        
        if stored_in_vdb:
            location_str = f"VectorDB (Chroma) -> Collection: 'context_offloads', ID: '{doc_id}'"
        else:
            try:
                with fname.open("w", encoding="utf-8") as fh:
                    json.dump({
                        "index": idx,
                        "role": role,
                        "sanitized": sanitized,
                        "original_length": len(content),
                        "tags": tags,
                    }, fh, indent=2, ensure_ascii=False)
                location_str = str(fname)
            except Exception:
                location_str = "Memory offload failed"

        # 3. Create Semantic Pointer
        pointer_desc = []
        if tags.get("actions"): pointer_desc.append("Actions: " + tags["actions"][0][:50])
        if tags.get("decisions"): pointer_desc.append("Decisions: " + tags["decisions"][0][:50])
        
        summary = " | ".join(pointer_desc) if pointer_desc else "Task execution details"
        
        pointer = f"[[OFFLOADED_TASK_POINTER]]\n" \
                  f"Role: {role}\n" \
                  f"Status: Completed/Archived\n" \
                  f"Summary: {summary}\n" \
                  f"Location: {location_str}\n" \
                  f"Note: This context was compressed. To retrieve, read JSON file, or query Vector DB."
        
        compacted.append({
            "role": role,
            "content": pointer
        })

    # Add back the recent active messages
    compacted.extend(keep_active)
    return compacted
