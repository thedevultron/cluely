"""Main-module tests: prompt/masking helpers + --diag/--auto argparse routing.

Importing main.py must NOT launch any GUI: PySide6 widgets are only imported
inside functions, and the entry point is behind the ``__main__`` guard.
These tests run headless (QT_QPA_PLATFORM=offscreen is set before PySide6).
"""
from __future__ import annotations

import os
import sys

# Must be set before any PySide6 import so offscreen widget creation works.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import main  # noqa: E402


def test_import_main_does_not_launch_gui():
    """`import main` is side-effect free w.r.t. Qt: no QApplication exists yet."""
    from PySide6.QtWidgets import QApplication

    assert QApplication.instance() is None


def test_difficulty_prompts_shape():
    assert set(main.DIFFICULTY_PROMPTS) == {"EASY", "MEDIUM", "HARD", "EXPERT"}
    for level, prompt in main.DIFFICULTY_PROMPTS.items():
        assert isinstance(prompt, str) and prompt.strip()
        assert "Difficulty:" in prompt, f"{level} prompt missing 'Difficulty:' marker"


def test_mask_key_empty():
    assert main._mask_key("") == "(empty)"
    assert main._mask_key("   ") == "(empty)"


def test_mask_key_short():
    masked = main._mask_key("abc")
    assert "*" in masked
    assert masked.startswith("a")
    assert len(masked) < 6


def test_mask_key_long_never_leaks_full_key():
    key = "AIzaSyABCDEFGHIJKLMNOP1234567890"
    masked = main._mask_key(key)
    assert key not in masked
    assert key[:4] in masked and key[-2:] in masked
    assert f"len={len(key)}" in masked


def _base_config(**overrides):
    cfg = {
        "system_instruction": "BASE",
        "audio_mode": "system",
        "difficulty_level": "MEDIUM",
        "target_role": "",
        "target_company": "",
    }
    cfg.update(overrides)
    return cfg


def test_enrich_instruction_adds_speaker_protocol_and_difficulty():
    out = main._enrich_instruction(_base_config())
    assert out.startswith("BASE")
    assert "Speaker protocol" in out
    assert "Difficulty:" in out
    assert "MEDIUM" in out


def test_enrich_instruction_is_idempotent():
    cfg = _base_config()
    once = main._enrich_instruction(cfg)
    twice = main._enrich_instruction({"system_instruction": once, **{k: v for k, v in cfg.items() if k != "system_instruction"}})
    assert twice == once


def test_enrich_instruction_unknown_difficulty_falls_back_to_medium():
    out = main._enrich_instruction(_base_config(difficulty_level="BOGUS"))
    assert "MEDIUM" in out


def test_enrich_instruction_dual_mode_uses_interviewer_candidate_protocol():
    out = main._enrich_instruction(_base_config(audio_mode="dual"))
    assert "INTERVIEWER" in out
    assert "CANDIDATE" in out


def test_enrich_instruction_adds_candidate_context():
    out = main._enrich_instruction(
        _base_config(target_role="Backend Engineer", target_company="Acme")
    )
    assert "Candidate context:" in out
    assert "Backend Engineer" in out
    assert "Acme" in out


def test_argparse_diag_routes_to_run_diag(monkeypatch):
    monkeypatch.setattr(main, "run_diag", lambda: 99)
    assert main.main(["--diag"]) == 99


def test_argparse_auto_routes_to_hud_session(monkeypatch):
    monkeypatch.setattr(main, "load_config", lambda: {"api_key": "test-key"})
    monkeypatch.setattr(main, "run_hud_session", lambda config: 7)
    assert main.main(["--auto"]) == 7


def test_argparse_rejects_unknown_flag():
    with pytest.raises(SystemExit):
        main.main(["--does-not-exist"])


# ---------------------------------------------------------------------------
# Fast-reply defaults: FAST_PROMPTS, concise difficulty, <2500 total
# ---------------------------------------------------------------------------

def test_fast_prompts_present_with_word_limits():
    assert set(main.FAST_PROMPTS) == {"EASY", "MEDIUM", "HARD", "EXPERT"}
    easy = main.FAST_PROMPTS["EASY"]
    medium = main.FAST_PROMPTS["MEDIUM"]
    hard = main.FAST_PROMPTS["HARD"]
    expert = main.FAST_PROMPTS["EXPERT"]
    assert "1 line" in easy.lower() and "40" in easy
    assert "2 bullets" in medium.lower() and "80" in medium
    assert "3 bullets" in hard.lower() and "150" in hard
    assert "4 bullets" in expert.lower() and "250" in expert


def test_difficulty_prompts_concise_with_length_limits():
    assert set(main.DIFFICULTY_PROMPTS) == {"EASY", "MEDIUM", "HARD", "EXPERT"}
    assert "Difficulty:" in main.DIFFICULTY_PROMPTS["EASY"]
    assert "40" in main.DIFFICULTY_PROMPTS["EASY"]
    assert "80" in main.DIFFICULTY_PROMPTS["MEDIUM"]
    assert "150" in main.DIFFICULTY_PROMPTS["HARD"]
    assert "250" in main.DIFFICULTY_PROMPTS["EXPERT"]
    # concise: each difficulty override is short (<120 chars) for fast tokens
    for level, prompt in main.DIFFICULTY_PROMPTS.items():
        assert 0 < len(prompt) < 120, f"{level} prompt not concise: {len(prompt)}"


def test_enrich_instruction_contains_fast_reply_directive():
    out = main._enrich_instruction(_base_config())
    assert "Reply within 2s" in out
    assert "first token immediate" in out
    assert "no preamble" in out
    assert "no hedging" in out
    assert "number/bullet only" in out


def test_enrich_instruction_total_length_under_2500_default():
    import config_manager as _cm

    cfg = {
        "system_instruction": _cm.DEFAULT_SYSTEM_INSTRUCTION,
        "audio_mode": "system",
        "difficulty_level": "EASY",
        "target_role": "",
        "target_company": "",
    }
    out = main._enrich_instruction(cfg)
    assert len(out) < 2500, f"enriched length {len(out)} >= 2500"


def test_enrich_instruction_truncates_long_input_under_2500():
    cfg = _base_config(system_instruction="X" * 5000)
    out = main._enrich_instruction(cfg)
    assert len(out) < 2500, f"long-input length {len(out)} >= 2500"


def test_enrich_instruction_fast_marker_idempotent():
    cfg = _base_config()
    once = main._enrich_instruction(cfg)
    twice_cfg = {"system_instruction": once, **{k: v for k, v in cfg.items() if k != "system_instruction"}}
    twice = main._enrich_instruction(twice_cfg)
    assert twice == once, "enrich is not idempotent"
    assert once.count("Reply within 2s") == 1
    assert twice.count("Reply within 2s") == 1


def test_no_1008_in_config_defaults():
    import config_manager as _cm

    assert "1008" not in repr(_cm.DEFAULT_CONFIG)
    out = main._enrich_instruction(_base_config())
    assert "1008" not in out


def test_stats_and_summarizer_intervals_tuned():
    import pathlib

    src = pathlib.Path(main.__file__).read_text(encoding="utf-8")
    assert "5_000" in src, "stats ticker should be 5s"
    assert "600_000" in src, "summarizer should be 600s"
    assert "10_000" not in src, "old 10s stats interval should be gone"
    assert "300_000" not in src, "old 300s summarizer interval should be gone"
