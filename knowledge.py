"""Local knowledge-base loader for InterviewCopilot.

Reads text content from user-configured files/directories and concatenates
it (truncated) for injection into the model system instruction.
Never raises: all I/O errors are swallowed and yield partial/empty output.
"""

from __future__ import annotations

import os

TEXT_EXTENSIONS = {".txt", ".md", ".py", ".json", ".csv", ".pdf", ".log", ".rst", ".toml", ".yaml", ".yml", ".ini", ".cfg"}
MAX_FILE_BYTES = 1024 * 1024  # skip files > 1MB


def list_knowledge_files(paths: list[str]) -> list[str]:
    """Return list of readable candidate files found under *paths*.

    *paths* may contain files and/or directories. Directories are walked
    recursively; only TEXT_EXTENSIONS are included. Never raises.
    """
    found: list[str] = []
    try:
        items = list(paths or [])
    except Exception:
        return []
    for raw in items:
        try:
            p = str(raw).strip()
            if not p:
                continue
            if os.path.isfile(p):
                ext = os.path.splitext(p)[1].lower()
                # skip obvious binaries even when added explicitly
                if ext and ext not in TEXT_EXTENSIONS and ext not in (".docx", ".doc"):
                    # allow extensionless files (often plain text), skip known binary
                    if ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".exe", ".dll",
                               ".zip", ".mp3", ".mp4", ".wav", ".pyc", ".so", ".pyd"):
                        continue
                found.append(os.path.abspath(p))
            elif os.path.isdir(p):
                for root, _dirs, files in os.walk(p):
                    for name in files:
                        try:
                            ext = os.path.splitext(name)[1].lower()
                            if ext in TEXT_EXTENSIONS:
                                full = os.path.join(root, name)
                                found.append(os.path.abspath(full))
                        except Exception:
                            continue
        except Exception:
            continue
    # de-dup preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for f in found:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


def _read_text_file(path: str) -> str | None:
    """Read a single file as utf-8 (errors=ignore). Returns None to skip."""
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            # Only try pypdf if installed; otherwise skip silently.
            try:
                from pypdf import PdfReader  # type: ignore
            except Exception:
                return None
            try:
                reader = PdfReader(path)
                chunks: list[str] = []
                for page in reader.pages:
                    try:
                        chunks.append(page.extract_text() or "")
                    except Exception:
                        continue
                text = "\n".join(chunks)
                return text if text.strip() else None
            except Exception:
                return None
        if ext in (".docx", ".doc"):
            return None  # binary office format — skip instead of mojibake
        try:
            if os.path.getsize(path) > MAX_FILE_BYTES:
                return None  # skip oversize files silently
        except OSError:
            return None
        # binary sniff: skip files with null bytes in first 8KB
        try:
            with open(path, "rb") as bf:
                head = bf.read(8192)
                if b"\x00" in head:
                    return None
        except Exception:
            pass
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return None


def load_knowledge(paths: list[str], max_chars: int) -> str:
    """Load and concatenate knowledge content, truncated to *max_chars*.

    Format per file: "\\n\\n===== FILE: <name> =====\\n<content>".
    Returns "" when nothing found. Never raises.
    """
    try:
        try:
            budget = int(max_chars)
        except (TypeError, ValueError):
            budget = 12000
        budget = max(1000, min(60000, budget))
    except Exception:
        budget = 12000
    try:
        files = list_knowledge_files(paths)
    except Exception:
        return ""
    parts: list[str] = []
    total = 0
    truncated = False
    for fp in files:
        if total >= budget:
            truncated = True
            break
        try:
            content = _read_text_file(fp)
            if content is None:
                continue  # binary/oversize/unreadable -> skip
            if not content.strip():
                continue
            name = os.path.basename(fp)
            header = f"\n\n===== FILE: {name} =====\n"
            # reserve header space
            room = budget - total - len(header)
            if room <= 0:
                truncated = True
                break
            chunk = content[:room]
            if len(content) > room:
                truncated = True
                chunk += "\n[…truncated to fit budget…]"
            parts.append(header + chunk)
            total += len(header) + len(chunk)
        except Exception:
            continue
    try:
        return "".join(parts)
    except Exception:
        return ""
