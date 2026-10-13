"""Thread-safe JSON config manager for InterviewCopilot."""

from __future__ import annotations

import copy
import json
import os
import threading
from typing import Any, Dict

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

DEFAULT_CONFIG: Dict[str, Any] = {
    "api_key": "",
    "audio_mode": "system",  # "system" = computer sound only (default), "dual" = mic + system
    "mic_enabled": False,
    "selected_mic_index": None,
    "selected_loopback_name": "",
    "vad_threshold_db": 300,
    "save_transcript_locally": True,
    "font_size": 14,
    "hud_opacity": 0.85,
    "model": "gemini-3.8-live",
    "fallback_model": "gemini-3.1-flash-live-preview",
    "ultimate_fallback_model": "gemini-2.5-flash-native-audio-preview-12-2025",
    "summarizer_model": "gemini-3.5-flash-lite",
    "thinking_level": "HIGH",
    "difficulty_level": "MEDIUM",  # user-facing difficulty: EASY/MEDIUM/HARD/EXPERT -> prompt mapping
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


def get_config_path() -> str:
    """Return writable config path (exe dir when frozen, else script dir)."""
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


def _merged_with_defaults(data: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(DEFAULT_CONFIG)
    for k, v in (data or {}).items():
        merged[k] = v
    return merged


def validate_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """Clamp / coerce values into safe ranges. Returns validated copy."""
    cfg = _merged_with_defaults(data)

    try:
        cfg["vad_threshold_db"] = int(cfg.get("vad_threshold_db", 300))
    except (TypeError, ValueError):
        cfg["vad_threshold_db"] = 300
    cfg["vad_threshold_db"] = max(50, min(2000, cfg["vad_threshold_db"]))

    try:
        cfg["font_size"] = int(cfg.get("font_size", 14))
    except (TypeError, ValueError):
        cfg["font_size"] = 14
    cfg["font_size"] = max(8, min(28, cfg["font_size"]))

    try:
        cfg["hud_opacity"] = float(cfg.get("hud_opacity", 0.85))
    except (TypeError, ValueError):
        cfg["hud_opacity"] = 0.85
    cfg["hud_opacity"] = max(0.3, min(1.0, cfg["hud_opacity"]))

    if cfg.get("selected_mic_index") is not None:
        try:
            cfg["selected_mic_index"] = int(cfg["selected_mic_index"])
        except (TypeError, ValueError):
            cfg["selected_mic_index"] = None

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
    if mode == "system":
        cfg["mic_enabled"] = False
    else:
        cfg["mic_enabled"] = True

    for key in ("api_key", "selected_loopback_name", "model",
                "fallback_model", "ultimate_fallback_model",
                "summarizer_model", "thinking_level", "difficulty_level",
                "target_role", "target_company", "system_instruction"):
        if cfg.get(key) is None:
            cfg[key] = DEFAULT_CONFIG[key]
        else:
            cfg[key] = str(cfg[key])
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
        if diff in ("EASY", "BEGINNER", "MINIMAL", "LOW"):
            diff = "EASY" if diff in ("EASY", "BEGINNER") else diff
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

    cfg["save_transcript_locally"] = bool(cfg.get("save_transcript_locally", True))

    cfg["enable_knowledge"] = bool(cfg.get("enable_knowledge", False))
    cfg["enable_web_search"] = bool(cfg.get("enable_web_search", False))
    cfg["hud_auto_resize"] = bool(cfg.get("hud_auto_resize", True))
    cfg["stealth_hide_from_capture"] = bool(cfg.get("stealth_hide_from_capture", True))
    cfg["selftest_hello"] = bool(cfg.get("selftest_hello", True))

    raw_paths = cfg.get("knowledge_paths", [])
    if isinstance(raw_paths, str):
        raw_paths = [raw_paths]
    if not isinstance(raw_paths, (list, tuple)):
        raw_paths = []
    cfg["knowledge_paths"] = [str(p) for p in raw_paths if str(p).strip()]

    try:
        cfg["knowledge_max_chars"] = int(cfg.get("knowledge_max_chars", 12000))
    except (TypeError, ValueError):
        cfg["knowledge_max_chars"] = 12000
    cfg["knowledge_max_chars"] = max(1000, min(60000, cfg["knowledge_max_chars"]))

    try:
        cfg["web_max_results"] = int(cfg.get("web_max_results", 5))
    except (TypeError, ValueError):
        cfg["web_max_results"] = 5
    cfg["web_max_results"] = max(1, min(10, cfg["web_max_results"]))

    return cfg


def load_config(path: str | None = None) -> Dict[str, Any]:
    """Load config from disk, merging defaults on missing/corrupt file."""
    with _lock:
        cfg_path = path or get_config_path()
        if not os.path.exists(cfg_path):
            return copy.deepcopy(DEFAULT_CONFIG)
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return copy.deepcopy(DEFAULT_CONFIG)
            return validate_config(data)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return copy.deepcopy(DEFAULT_CONFIG)


def save_config(data: Dict[str, Any], path: str | None = None) -> str:
    """Validate and atomically write config. Returns path written."""
    with _lock:
        cfg_path = path or get_config_path()
        validated = validate_config(data)
        tmp_path = cfg_path + ".tmp"
        try:
            os.makedirs(os.path.dirname(cfg_path) or ".", exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(validated, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, cfg_path)
        except OSError:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            raise
        return cfg_path


def update_config(partial: Dict[str, Any], path: str | None = None) -> Dict[str, Any]:
    """Merge partial dict into stored config and persist."""
    with _lock:
        current = load_config(path)
        current.update(partial or {})
        validated = validate_config(current)
        save_config(validated, path)
        return validated


def reset_to_defaults(path: str | None = None) -> Dict[str, Any]:
    with _lock:
        save_config(copy.deepcopy(DEFAULT_CONFIG), path)
        return copy.deepcopy(DEFAULT_CONFIG)
