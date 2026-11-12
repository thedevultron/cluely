"""Thread-safe JSON config manager for InterviewCopilot.

Stores user settings in ``config.json`` next to the script (or next to the
frozen executable). The file contains a live API key: it is listed in
``.gitignore`` and must never be committed (redacted in logs/diagnostics).

Public API is exported via ``__all__``. All public functions are safe to
call from multiple threads and never raise on bad input except
:func:`save_config`, which re-raises :class:`OSError` /
:class:`TypeError` / :class:`ValueError` after cleaning up its temp file.
"""

from __future__ import annotations

import copy
import json
import math
import os
import threading
from typing import Any

__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_SYSTEM_INSTRUCTION",
    "OLD_DEFAULT_SYSTEM_INSTRUCTION",
    "PREV_DEFAULT_SYSTEM_INSTRUCTION_V1",
    "get_config_path",
    "load_config",
    "reset_to_defaults",
    "save_config",
    "update_config",
    "validate_config",
]

OLD_DEFAULT_SYSTEM_INSTRUCTION = (
    "You are an elite, invisible real-time technical interview copilot.\n"
    "The input audio combines two speakers:\n"
    "1. The Interviewer (audio through system loopback, asking questions, stating problem statements).\n"
    "2. The Candidate (audio through the direct mic).\n"
    "\n"
    "Operational Directives:\n"
    "1. SILENCE DISCIPLINE: If the Candidate is speaking, explaining an approach, or engaging in small talk, emit ABSOLUTELY NOTHING. Do not generate conversational fillers or acknowledge Candidate speech.\n"
    "2. QUESTION TRIGGER: When the Interviewer asks a question, poses a coding challenge, or presents a system architecture problem, respond immediately with high-density, concise guidance.\n"
    "3. OUTPUT FORMAT:\n"
    "   - Provide responses in maximum 3-5 bullet points.\n"
    "   - Core Algorithm & Pattern: (e.g., Two Pointers, Monotonic Stack, Backtracking, Raft Consensus).\n"
    "   - Optimal Time & Space Complexity: (e.g., O(N log N) time, O(1) auxiliary space).\n"
    "   - Key Edge Cases: (e.g., empty string, overflow, negative values, cyclic graphs).\n"
    "   - Pseudocode / Code Sketch: Provide only 2-4 critical lines of syntax or idiomatic code if an exact tricky logic step is required.\n"
    "4. TONE: Extreme brevity, zero fluff, objective technical hints only."
)

# v1 system-only default (shipped briefly; lacks recency/brevity directives).
PREV_DEFAULT_SYSTEM_INSTRUCTION_V1 = (
    "You are an elite real-time technical interview copilot.\n"
    "You hear ONLY the computer's system audio: the interviewer, the meeting,\n"
    "or whatever plays through the computer. There is no microphone and no\n"
    "second speaker channel — everything you hear is one shared audio stream.\n"
    "\n"
    "Operational Directives:\n"
    "1. ANSWER TRIGGER: When you hear a question, coding challenge, or system design problem, respond immediately with high-density, concise guidance.\n"
    "2. SILENCE: If you hear only greetings, small talk, or non-technical chatter, emit ABSOLUTELY NOTHING.\n"
    "3. OUTPUT FORMAT:\n"
    "   - Maximum 3-5 bullet points.\n"
    "   - Core Algorithm & Pattern: (e.g., Two Pointers, Monotonic Stack, Backtracking, Raft Consensus).\n"
    "   - Optimal Time & Space Complexity: (e.g., O(N log N) time, O(1) auxiliary space).\n"
    "   - Key Edge Cases: (e.g., empty string, overflow, negative values, cyclic graphs).\n"
    "   - Pseudocode / Code Sketch: 2-4 critical lines only when an exact tricky step is required.\n"
    "4. TONE: Extreme brevity, zero fluff, objective technical hints only.\n"
    "5. TEXT ONLY: Your reply is shown as text. Never request or produce audio."
)

DEFAULT_SYSTEM_INSTRUCTION = (
    "You are an elite real-time technical interview copilot.\n"
    "You hear ONLY the computer's system audio: the interviewer, the meeting,\n"
    "or whatever plays through the computer. There is no microphone and no\n"
    "second speaker channel — everything you hear is one shared audio stream.\n"
    "\n"
    "Operational Directives:\n"
    "1. ANSWER TRIGGER: When you hear a question, coding challenge, or system design problem, respond immediately with high-density, concise guidance.\n"
    "2. RECENCY: Always answer the MOST RECENT question. Ignore older questions already answered.\n"
    "3. BREVITY FOR SPEED: 1-3 short bullets/sentences, no preamble, no filler. Start directly with the answer.\n"
    "4. SILENCE: If you hear only greetings, small talk, or non-technical chatter, emit ABSOLUTELY NOTHING.\n"
    "5. OUTPUT FORMAT:\n"
    "   - Maximum 3-5 bullet points.\n"
    "   - Core Algorithm & Pattern: (e.g., Two Pointers, Monotonic Stack, Backtracking, Raft Consensus).\n"
    "   - Optimal Time & Space Complexity: (e.g., O(N log N) time, O(1) auxiliary space).\n"
    "   - Key Edge Cases: (e.g., empty string, overflow, negative values, cyclic graphs).\n"
    "   - Pseudocode / Code Sketch: 2-4 critical lines only when an exact tricky step is required.\n"
    "6. TONE: Extreme brevity, zero fluff, objective technical hints only.\n"
    "7. TEXT ONLY: Your reply is shown as text. Never request or produce audio."
)

DEFAULT_CONFIG: dict[str, Any] = {
    "api_key": "",
    "audio_mode": "system",  # "system" = computer sound only (default), "dual" = mic + system
    "mic_enabled": False,
    "selected_mic_index": None,
    "selected_loopback_name": "",
    "vad_threshold_db": 120,
    "save_transcript_locally": True,
    "font_size": 14,
    "hud_opacity": 0.85,
    "model": "gemini-3.8-live",
    "fallback_model": "gemini-3.1-flash-live-preview",
    "ultimate_fallback_model": "gemini-2.5-flash-native-audio-preview-12-2025",
    "summarizer_model": "gemini-3.5-flash-lite",
    "thinking_level": "MINIMAL",
    "difficulty_level": "EASY",  # user-facing difficulty: EASY/MEDIUM/HARD/EXPERT -> prompt mapping
    "target_role": "",
    "target_company": "",
    "system_instruction": DEFAULT_SYSTEM_INSTRUCTION,
    "enable_knowledge": False,
    "knowledge_paths": [],
    "knowledge_max_chars": 12000,
    "enable_web_search": False,
    "web_max_results": 5,
    "hud_auto_resize": True,
    "stealth_hide_from_capture": True,
    "selftest_hello": True,  # on first connect, ask model for a one-line reply (proves text path)
}

_lock = threading.RLock()

# String keys for which an empty/whitespace value is invalid and must
# fall back to the default (an empty model name would fail downstream).
_EMPTY_FALLBACK_KEYS = frozenset({
    "model",
    "fallback_model",
    "ultimate_fallback_model",
    "summarizer_model",
    "system_instruction",
})

_FALSE_STRINGS = frozenset({"", "false", "0", "no", "off", "n", "f"})
_TRUE_STRINGS = frozenset({"true", "1", "yes", "on", "y", "t"})


def get_config_path() -> str:
    """Return the writable config path (exe dir when frozen, else script dir)."""
    try:
        import sys as _sys

        if getattr(_sys, "frozen", False):
            base = os.path.dirname(os.path.abspath(_sys.executable))
        else:
            base = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base = os.getcwd()
    except Exception:
        try:
            base = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base = os.getcwd()
    return os.path.join(base, "config.json")


def _coerce_int(value: Any, default: int, low: int, high: int) -> int:
    """Coerce *value* to an int in ``[low, high]``; bad input returns *default*."""
    if isinstance(value, bool):
        return default
    try:
        n = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return default
    return max(low, min(high, n))


def _coerce_float(value: Any, default: float, low: float, high: float) -> float:
    """Coerce *value* to a finite float in ``[low, high]``."""
    if isinstance(value, bool):
        return default
    try:
        n = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(n):
        return default
    return max(low, min(high, n))


def _coerce_bool(value: Any, default: bool) -> bool:
    """Coerce *value* to a bool, understanding common string spellings."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        s = value.strip().lower()
        if s in _FALSE_STRINGS:
            return False
        if s in _TRUE_STRINGS:
            return True
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return bool(value)


def _merged_with_defaults(data: Any) -> dict[str, Any]:
    """Merge *data* over a deep copy of :data:`DEFAULT_CONFIG`."""
    merged = copy.deepcopy(DEFAULT_CONFIG)
    if not isinstance(data, dict):
        return merged
    for k, v in data.items():
        if not isinstance(k, str):
            continue
        if k.startswith("_"):
            continue  # transient/private state (e.g. _knowledge_summary) is never persisted
        merged[k] = v
    return merged


def validate_config(data: dict[str, Any]) -> dict[str, Any]:
    """Clamp / coerce values into safe ranges. Returns a validated copy.

    Never raises on bad input: non-dict input is treated as empty, bools are
    not accepted as ints/floats, NaN/Inf fall back to defaults, and model
    fields with empty values fall back to their defaults. Unknown public
    keys are preserved for forward compatibility; ``_``-prefixed keys are
    stripped.
    """
    cfg = _merged_with_defaults(data)

    cfg["vad_threshold_db"] = _coerce_int(cfg.get("vad_threshold_db", 300), 300, 50, 2000)
    cfg["font_size"] = _coerce_int(cfg.get("font_size", 14), 14, 8, 28)
    cfg["hud_opacity"] = _coerce_float(cfg.get("hud_opacity", 0.85), 0.85, 0.3, 1.0)

    mic = cfg.get("selected_mic_index")
    if mic is None:
        cfg["selected_mic_index"] = None
    elif isinstance(mic, bool):
        cfg["selected_mic_index"] = None
    else:
        try:
            mic_int = int(mic)  # type: ignore[arg-type]
        except (TypeError, ValueError, OverflowError):
            cfg["selected_mic_index"] = None
        else:
            cfg["selected_mic_index"] = mic_int if mic_int >= 0 else None

    # audio mode: system-only (computer sound) is the default; dual adds mic
    try:
        mode = str(cfg.get("audio_mode", "system") or "system").strip().lower()
    except Exception:
        mode = "system"
    if mode not in ("system", "dual", "loopback", "computer"):
        mode = "system"
    if mode in ("loopback", "computer"):
        mode = "system"
    cfg["audio_mode"] = mode
    # mic_enabled is derived strictly from audio_mode (system = computer only)
    cfg["mic_enabled"] = mode != "system"

    for key in ("api_key", "selected_loopback_name", "model",
                "fallback_model", "ultimate_fallback_model",
                "summarizer_model", "thinking_level", "difficulty_level",
                "target_role", "target_company", "system_instruction"):
        value = cfg.get(key)
        if value is None:
            cfg[key] = DEFAULT_CONFIG[key]
        elif key in _EMPTY_FALLBACK_KEYS and isinstance(value, str) and not value.strip():
            cfg[key] = DEFAULT_CONFIG[key]
        else:
            cfg[key] = str(value)
    # migrate retired default prompts to the current one.
    # Custom user prompts are never touched.
    try:
        _si = str(cfg.get("system_instruction", "")).strip()
        if _si == OLD_DEFAULT_SYSTEM_INSTRUCTION.strip():
            cfg["system_instruction"] = DEFAULT_SYSTEM_INSTRUCTION
        elif _si == PREV_DEFAULT_SYSTEM_INSTRUCTION_V1.strip():
            cfg["system_instruction"] = DEFAULT_SYSTEM_INSTRUCTION
    except Exception:
        pass
    # normalize thinking level to SDK enum names
    try:
        lvl = str(cfg.get("thinking_level", "HIGH")).strip().upper()
        if lvl in ("EXTENDED", "MAX", "ULTRA"):
            lvl = "HIGH"
        if lvl not in ("MINIMAL", "LOW", "MEDIUM", "HIGH"):
            lvl = "HIGH"
        cfg["thinking_level"] = lvl
    except Exception:
        cfg["thinking_level"] = "HIGH"
    # normalize difficulty (user-facing)
    try:
        diff = str(cfg.get("difficulty_level", "MEDIUM") or "MEDIUM").strip().upper()
        aliases = {"BEGINNER": "EASY", "NORMAL": "MEDIUM", "HARD": "HARD",
                   "EXPERT": "EXPERT", "MINIMAL": "EASY", "LOW": "EASY",
                   "MEDIUM": "MEDIUM", "HIGH": "HARD", "EXTENDED": "EXPERT",
                   "MAX": "EXPERT", "ULTRA": "EXPERT"}
        diff = aliases.get(diff, diff)
        if diff not in ("EASY", "MEDIUM", "HARD", "EXPERT"):
            diff = "MEDIUM"
        cfg["difficulty_level"] = diff
    except Exception:
        cfg["difficulty_level"] = "MEDIUM"

    cfg["save_transcript_locally"] = _coerce_bool(cfg.get("save_transcript_locally", True), True)
    cfg["enable_knowledge"] = _coerce_bool(cfg.get("enable_knowledge", False), False)
    cfg["enable_web_search"] = _coerce_bool(cfg.get("enable_web_search", False), False)
    cfg["hud_auto_resize"] = _coerce_bool(cfg.get("hud_auto_resize", True), True)
    cfg["stealth_hide_from_capture"] = _coerce_bool(
        cfg.get("stealth_hide_from_capture", True), True
    )
    cfg["selftest_hello"] = _coerce_bool(cfg.get("selftest_hello", True), True)

    raw_paths = cfg.get("knowledge_paths", [])
    if isinstance(raw_paths, (str, os.PathLike)):
        raw_paths = [str(raw_paths)]
    if not isinstance(raw_paths, (list, tuple)):
        raw_paths = []
    cleaned: list[str] = []
    seen: set[str] = set()
    for p in raw_paths:
        if p is None:
            continue
        s = str(p).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        cleaned.append(s)
    cfg["knowledge_paths"] = cleaned

    cfg["knowledge_max_chars"] = _coerce_int(
        cfg.get("knowledge_max_chars", 12000), 12000, 1000, 60000
    )
    cfg["web_max_results"] = _coerce_int(cfg.get("web_max_results", 5), 5, 1, 10)

    return cfg


def load_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load config from disk, merging defaults on missing/corrupt file."""
    with _lock:
        cfg_path = str(path) if path is not None else get_config_path()
        if not os.path.exists(cfg_path):
            return validate_config({})
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return validate_config({})
            return validate_config(data)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return validate_config({})


def save_config(data: dict[str, Any], path: str | os.PathLike[str] | None = None) -> str:
    """Validate and atomically write config. Returns the path written.

    Writes to a sibling ``.tmp`` file, fsyncs, then ``os.replace`` (atomic
    on Windows and POSIX). On failure the temp file is removed and the
    exception re-raised. The file may contain an API key: permissions are
    restricted best-effort (``0o600`` where supported).
    """
    with _lock:
        cfg_path = str(path) if path is not None else get_config_path()
        validated = validate_config(data)
        tmp_path = cfg_path + ".tmp"
        try:
            os.makedirs(os.path.dirname(cfg_path) or ".", exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(validated, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, cfg_path)
            try:
                os.chmod(cfg_path, 0o600)
            except (OSError, NotImplementedError):
                pass
        except (OSError, TypeError, ValueError):
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            raise
        return cfg_path


def update_config(
    partial: dict[str, Any], path: str | os.PathLike[str] | None = None
) -> dict[str, Any]:
    """Merge a partial dict into the stored config, validate, and persist."""
    with _lock:
        current = load_config(path)
        if isinstance(partial, dict):
            current.update(partial)
        validated = validate_config(current)
        save_config(validated, path)
        return validated


def reset_to_defaults(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Overwrite config on disk with validated defaults; returns the defaults."""
    with _lock:
        save_config(copy.deepcopy(DEFAULT_CONFIG), path)
        return validate_config({})
