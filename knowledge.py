"""Local knowledge-base loader for InterviewCopilot.

Reads text content from user-configured files/directories and concatenates
it (truncated) for injection into the model system instruction.
Never raises: all I/O errors are swallowed and yield partial/empty output.

Public API is exported via ``__all__``. ``knowledge_paths`` in config are
user-chosen local paths: absolute paths and ``..`` segments are allowed by
design (the user explicitly lists their own documents). Only text-like
extensions are walked; known binaries are skipped.
"""

from __future__ import annotations

import os

__all__ = [
    "MAX_FILE_BYTES",
    "TEXT_EXTENSIONS",
    "list_knowledge_files",
    "load_knowledge",
]

TEXT_EXTENSIONS: set[str] = {
    ".txt", ".md", ".py", ".json", ".csv", ".pdf", ".log",
    ".rst", ".toml", ".yaml", ".yml", ".ini", ".cfg",
}
MAX_FILE_BYTES: int = 1024 * 1024  # skip files > 1MB

# Extensions that are never treated as text, even when added explicitly.
_BINARY_EXTENSIONS: frozenset[str] = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp",
    ".exe", ".dll", ".zip",
    ".mp3", ".mp4", ".wav",
    ".pyc", ".so", ".pyd",
    ".bin", ".dat", ".class", ".o", ".a", ".obj", ".lib",
})

_TRUNC_MARKER: str = "\n[…truncated to fit budget…]"


def _normalize_paths(paths: object) -> list[str]:
    """Coerce *paths* into a list of path strings (handles a bare ``str``)."""
    if paths is None:
        return []
    if isinstance(paths, (str, os.PathLike)):
        return [str(paths)]
    try:
        items = list(paths)  # type: ignore[arg-type]
    except Exception:
        return []
    return items  # type: ignore[return-value]


def list_knowledge_files(
    paths: list[str] | tuple[str, ...] | str | os.PathLike[str] | None,
) -> list[str]:
    """Return a list of readable candidate files found under *paths*.

    *paths* may contain files and/or directories (a single path string is
    treated as one path, not iterated character by character). Directories
    are walked recursively; only :data:`TEXT_EXTENSIONS` are included.
    Explicitly listed files must not be known binaries; extensionless files
    are allowed (often plain text). Never raises.
    """
    found: list[str] = []
    for raw in _normalize_paths(paths):
        try:
            if raw is None:
                continue
            p = str(raw).strip()
            if not p:
                continue
            if os.path.isfile(p):
                ext = os.path.splitext(p)[1].lower()
                if ext in _BINARY_EXTENSIONS:
                    continue
                # allow extensionless files (often plain text), skip known binary
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
    # de-dup preserving order (case-insensitive on Windows)
    seen: set[str] = set()
    unique: list[str] = []
    for f in found:
        key = os.path.normcase(os.path.normpath(f))
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def _read_text_file(path: str) -> str | None:
    """Read a single file as utf-8 (errors=ignore). Returns None to skip."""
    p = str(path or "")
    if not p:
        return None
    try:
        try:
            if os.path.getsize(p) > MAX_FILE_BYTES:
                return None  # skip oversize files silently (applies to pdf too)
        except OSError:
            return None
        ext = os.path.splitext(p)[1].lower()
        if ext == ".pdf":
            # Only try pypdf if installed; otherwise skip silently.
            try:
                from pypdf import PdfReader  # type: ignore
            except Exception:
                return None
            try:
                reader = PdfReader(p)
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
        # binary sniff: skip files with null bytes in first 8KB
        try:
            with open(p, "rb") as bf:
                head = bf.read(8192)
                if b"\x00" in head:
                    return None
        except Exception:
            pass
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return None


def load_knowledge(
    paths: list[str] | tuple[str, ...] | str | os.PathLike[str] | None,
    max_chars: int,
) -> str:
    """Load and concatenate knowledge content, truncated to *max_chars*.

    Format per file: ``"\\n\\n===== FILE: <name> =====\\n<content>"``.
    The returned string never exceeds the clamped budget
    (1000–60000 chars) — the truncation marker is accounted for when
    slicing. Returns ``""`` when nothing is found. Never raises.
    """
    try:
        if isinstance(max_chars, bool):
            budget = 12000
        else:
            try:
                budget = int(max_chars)  # type: ignore[arg-type]
            except (TypeError, ValueError, OverflowError):
                budget = 12000
        budget = max(1000, min(60000, budget))
    except Exception:
        budget = 12000
    try:
        files = list_knowledge_files(paths)  # type: ignore[arg-type]
    except Exception:
        return ""
    parts: list[str] = []
    total = 0
    for fp in files:
        if total >= budget:
            break
        try:
            content = _read_text_file(fp)
            if content is None:
                continue  # binary/oversize/unreadable -> skip
            if not content.strip():
                continue
            name = os.path.basename(fp)
            header = f"\n\n===== FILE: {name} =====\n"
            if total + len(header) + len(content) <= budget:
                parts.append(header + content)
                total += len(header) + len(content)
            else:
                room = budget - total - len(header) - len(_TRUNC_MARKER)
                if room <= 0:
                    break
                chunk = content[:room] + _TRUNC_MARKER
                parts.append(header + chunk)
                total += len(header) + len(chunk)
                break  # budget exactly full; no room for further files
        except Exception:
            continue
    try:
        return "".join(parts)
    except Exception:
        return ""
