"""Tests for config_manager: stdlib + pytest only, no network, no real config.json."""

from __future__ import annotations

import copy
import json
import os
import threading

import pytest

import config_manager as cm


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _tmp(tmp_path):
    return str(tmp_path / "config.json")


# ---------------------------------------------------------------------------
# validate_config clamping / coercion
# ---------------------------------------------------------------------------

def test_validate_missing_keys_returns_defaults():
    cfg = cm.validate_config({})
    assert cfg["api_key"] == ""
    assert cfg["font_size"] == 14
    assert cfg["hud_opacity"] == 0.85
    assert cfg["audio_mode"] == "system"
    assert cfg["mic_enabled"] is False  # derived from audio_mode
    assert cfg["model"] == cm.DEFAULT_CONFIG["model"]
    assert cfg["web_max_results"] == 5
    assert cfg["knowledge_max_chars"] == 12000


def test_validate_none_and_non_dict_does_not_crash():
    # None / list / int / string inputs must not raise (validation gap fix)
    for bad in (None, [], ["x"], 42, "nope", 3.5, True):
        cfg = cm.validate_config(bad)  # type: ignore[arg-type]
        assert isinstance(cfg, dict)
        assert cfg["font_size"] == 14
        assert cfg["model"] == cm.DEFAULT_CONFIG["model"]


def test_validate_clamps_numeric_ranges():
    cfg = cm.validate_config({"vad_threshold_db": 99999, "font_size": 1,
                              "hud_opacity": 5.0, "knowledge_max_chars": 1,
                              "web_max_results": 99})
    assert cfg["vad_threshold_db"] == 2000
    assert cfg["font_size"] == 8
    assert cfg["hud_opacity"] == 1.0
    assert cfg["knowledge_max_chars"] == 1000
    assert cfg["web_max_results"] == 10

    cfg2 = cm.validate_config({"vad_threshold_db": 1, "font_size": 999,
                               "hud_opacity": -2.0, "knowledge_max_chars": 999999,
                               "web_max_results": 0})
    assert cfg2["vad_threshold_db"] == 50
    assert cfg2["font_size"] == 28
    assert cfg2["hud_opacity"] == 0.3
    assert cfg2["knowledge_max_chars"] == 60000
    assert cfg2["web_max_results"] == 1


def test_validate_bad_numeric_types_fall_back():
    cfg = cm.validate_config({"vad_threshold_db": "abc", "font_size": None,
                              "hud_opacity": "xyz", "knowledge_max_chars": "zz",
                              "web_max_results": object()})
    assert cfg["vad_threshold_db"] == 300
    assert cfg["font_size"] == 14
    assert cfg["hud_opacity"] == 0.85
    assert cfg["knowledge_max_chars"] == 12000
    assert cfg["web_max_results"] == 5


def test_validate_bool_is_not_valid_int():
    # bool is subclass of int: True would become 1 then clamped; must fall back
    cfg = cm.validate_config({"vad_threshold_db": True, "font_size": False,
                              "hud_opacity": True, "knowledge_max_chars": True,
                              "web_max_results": False})
    assert cfg["vad_threshold_db"] == 300
    assert cfg["font_size"] == 14
    assert cfg["hud_opacity"] == 0.85
    assert cfg["knowledge_max_chars"] == 12000
    assert cfg["web_max_results"] == 5


def test_validate_nan_inf_opacity_falls_back():
    import math
    for bad in (float("nan"), float("inf"), float("-inf"), "nan", "inf"):
        cfg = cm.validate_config({"hud_opacity": bad})
        assert math.isfinite(cfg["hud_opacity"])
        assert cfg["hud_opacity"] == 0.85


def test_validate_audio_mode_derives_mic_enabled():
    assert cm.validate_config({"audio_mode": "dual"})["mic_enabled"] is True
    assert cm.validate_config({"audio_mode": "DUAL"})["audio_mode"] == "dual"
    assert cm.validate_config({"audio_mode": "bogus"})["audio_mode"] == "system"
    assert cm.validate_config({"audio_mode": "loopback"})["audio_mode"] == "system"
    assert cm.validate_config({"audio_mode": "computer"})["audio_mode"] == "system"
    assert cm.validate_config({"audio_mode": None})["audio_mode"] == "system"
    # mic_enabled in input is ignored / derived
    assert cm.validate_config({"audio_mode": "system", "mic_enabled": True})["mic_enabled"] is False


def test_validate_selected_mic_index():
    assert cm.validate_config({"selected_mic_index": "3"})["selected_mic_index"] == 3
    assert cm.validate_config({"selected_mic_index": 2})["selected_mic_index"] == 2
    assert cm.validate_config({"selected_mic_index": None})["selected_mic_index"] is None
    assert cm.validate_config({"selected_mic_index": "abc"})["selected_mic_index"] is None
    assert cm.validate_config({"selected_mic_index": -1})["selected_mic_index"] is None  # negative invalid
    assert cm.validate_config({"selected_mic_index": True})["selected_mic_index"] is None  # bool invalid


def test_validate_bool_strings():
    cfg = cm.validate_config({"enable_knowledge": "false",
                              "enable_web_search": "0",
                              "hud_auto_resize": "no",
                              "stealth_hide_from_capture": "off",
                              "save_transcript_locally": ""})
    assert cfg["enable_knowledge"] is False
    assert cfg["enable_web_search"] is False
    assert cfg["hud_auto_resize"] is False
    assert cfg["stealth_hide_from_capture"] is False
    assert cfg["save_transcript_locally"] is False
    cfg2 = cm.validate_config({"enable_knowledge": "true",
                               "enable_web_search": "1",
                               "hud_auto_resize": "yes"})
    assert cfg2["enable_knowledge"] is True
    assert cfg2["enable_web_search"] is True
    assert cfg2["hud_auto_resize"] is True


def test_validate_model_empty_falls_back():
    cfg = cm.validate_config({"model": "", "fallback_model": "  ",
                              "ultimate_fallback_model": None,
                              "summarizer_model": "", "system_instruction": ""})
    assert cfg["model"] == cm.DEFAULT_CONFIG["model"]
    assert cfg["fallback_model"] == cm.DEFAULT_CONFIG["fallback_model"]
    assert cfg["ultimate_fallback_model"] == cm.DEFAULT_CONFIG["ultimate_fallback_model"]
    assert cfg["summarizer_model"] == cm.DEFAULT_CONFIG["summarizer_model"]
    assert cfg["system_instruction"] == cm.DEFAULT_SYSTEM_INSTRUCTION


def test_validate_preserves_custom_system_instruction():
    cfg = cm.validate_config({"system_instruction": "my custom prompt"})
    assert cfg["system_instruction"] == "my custom prompt"
    # retired defaults migrate
    cfg_old = cm.validate_config({"system_instruction": cm.OLD_DEFAULT_SYSTEM_INSTRUCTION})
    assert cfg_old["system_instruction"] == cm.DEFAULT_SYSTEM_INSTRUCTION


def test_validate_thinking_and_difficulty_normalization():
    assert cm.validate_config({"thinking_level": "extended"})["thinking_level"] == "HIGH"
    assert cm.validate_config({"thinking_level": "bogus"})["thinking_level"] == "HIGH"
    assert cm.validate_config({"thinking_level": "low"})["thinking_level"] == "LOW"
    assert cm.validate_config({"difficulty_level": "beginner"})["difficulty_level"] == "EASY"
    assert cm.validate_config({"difficulty_level": "bogus"})["difficulty_level"] == "MEDIUM"
    assert cm.validate_config({"difficulty_level": "expert"})["difficulty_level"] == "EXPERT"


def test_validate_knowledge_paths():
    cfg = cm.validate_config({"knowledge_paths": "/a/b.txt"})
    assert cfg["knowledge_paths"] == ["/a/b.txt"]
    cfg2 = cm.validate_config({"knowledge_paths": [None, "", "  ", "/x.txt", 123, "/x.txt", ""]})
    # None/blank dropped, numbers stringified, dupes removed
    assert "/x.txt" in cfg2["knowledge_paths"]
    assert "123" in cfg2["knowledge_paths"]
    assert "None" not in cfg2["knowledge_paths"]
    assert "" not in cfg2["knowledge_paths"]
    assert len(cfg2["knowledge_paths"]) == len(set(cfg2["knowledge_paths"]))
    assert cm.validate_config({"knowledge_paths": 123})["knowledge_paths"] == []


def test_validate_strips_private_underscore_keys():
    cfg = cm.validate_config({"_knowledge_summary": "secret-ish", "font_size": 20})
    assert "_knowledge_summary" not in cfg
    assert cfg["font_size"] == 20
    # unknown public keys preserved for backward compatibility
    cfg2 = cm.validate_config({"some_future_key": 1})
    assert cfg2["some_future_key"] == 1


def test_validate_does_not_mutate_input():
    data = {"font_size": 20, "knowledge_paths": ["/a"]}
    snapshot = copy.deepcopy(data)
    cm.validate_config(data)
    assert data == snapshot


# ---------------------------------------------------------------------------
# load / save / update / reset
# ---------------------------------------------------------------------------

def test_load_missing_file_returns_defaults(tmp_path):
    cfg = cm.load_config(_tmp(tmp_path))
    assert cfg == cm.validate_config({}) or cfg["font_size"] == 14
    assert cfg["audio_mode"] == "system"


def test_load_empty_and_corrupt_and_non_dict_json(tmp_path):
    p = _tmp(tmp_path)
    for payload in ("", "not json {{{", "null", "[]", "123", '"str"'):
        with open(p, "w", encoding="utf-8") as f:
            f.write(payload)
        cfg = cm.load_config(p)
        assert isinstance(cfg, dict)
        assert cfg["font_size"] == 14


def test_load_binary_garbage_returns_defaults(tmp_path):
    p = _tmp(tmp_path)
    with open(p, "wb") as f:
        f.write(b"\x00\xff\xfe garbage")
    cfg = cm.load_config(p)
    assert cfg["font_size"] == 14


def test_save_and_load_roundtrip(tmp_path):
    p = _tmp(tmp_path)
    data = {"api_key": "test-key-123", "font_size": 18, "hud_opacity": 0.5,
            "audio_mode": "dual", "knowledge_paths": ["/tmp/a.txt"]}
    written = cm.save_config(data, p)
    assert written == p
    assert os.path.exists(p)
    assert not os.path.exists(p + ".tmp")  # atomic: no tmp left behind
    loaded = cm.load_config(p)
    assert loaded["api_key"] == "test-key-123"
    assert loaded["font_size"] == 18
    assert loaded["hud_opacity"] == 0.5
    assert loaded["audio_mode"] == "dual"
    assert loaded["mic_enabled"] is True
    assert loaded["knowledge_paths"] == ["/tmp/a.txt"]


def test_save_validates_and_clamps(tmp_path):
    p = _tmp(tmp_path)
    cm.save_config({"font_size": 999, "web_max_results": 0}, p)
    loaded = cm.load_config(p)
    assert loaded["font_size"] == 28
    assert loaded["web_max_results"] == 1


def test_save_does_not_persist_private_keys(tmp_path):
    p = _tmp(tmp_path)
    cm.save_config({"_knowledge_summary": "x", "font_size": 16}, p)
    with open(p, encoding="utf-8") as f:
        on_disk = json.load(f)
    assert "_knowledge_summary" not in on_disk
    assert on_disk["font_size"] == 16


def test_save_nonserializable_cleans_tmp(tmp_path):
    p = _tmp(tmp_path)
    bad = {"font_size": 14, "oops": object()}  # validate keeps it, json.dump fails
    with pytest.raises((TypeError, ValueError)):
        cm.save_config(bad, p)
    assert not os.path.exists(p + ".tmp")


def test_save_creates_parent_dirs(tmp_path):
    p = str(tmp_path / "nested" / "deep" / "config.json")
    cm.save_config({"font_size": 12}, p)
    assert os.path.exists(p)


def test_update_config_merges(tmp_path):
    p = _tmp(tmp_path)
    cm.save_config({"font_size": 14, "hud_opacity": 0.9}, p)
    out = cm.update_config({"font_size": 20}, p)
    assert out["font_size"] == 20
    assert out["hud_opacity"] == 0.9  # preserved
    loaded = cm.load_config(p)
    assert loaded["font_size"] == 20


def test_update_config_bad_partial_does_not_crash(tmp_path):
    p = _tmp(tmp_path)
    cm.save_config({"font_size": 15}, p)
    for bad in (None, [], "x", 42):
        out = cm.update_config(bad, p)  # type: ignore[arg-type]
        assert out["font_size"] == 15


def test_reset_to_defaults(tmp_path):
    p = _tmp(tmp_path)
    cm.save_config({"font_size": 22, "model": "custom-model"}, p)
    out = cm.reset_to_defaults(p)
    assert out["font_size"] == 14
    assert out["model"] == cm.DEFAULT_CONFIG["model"]
    assert cm.load_config(p)["font_size"] == 14


def test_concurrent_save_load_no_crash(tmp_path):
    p = _tmp(tmp_path)
    errors: list[BaseException] = []

    def worker(n: int) -> None:
        try:
            for i in range(30):
                cm.save_config({"font_size": 10 + (i % 10)}, p)
                cfg = cm.load_config(p)
                assert 8 <= cfg["font_size"] <= 28
        except BaseException as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    assert not errors


def test_all_exports():
    assert hasattr(cm, "__all__")
    for name in ("DEFAULT_CONFIG", "validate_config", "load_config",
                 "save_config", "update_config", "reset_to_defaults",
                 "get_config_path"):
        assert name in cm.__all__


# ---------------------------------------------------------------------------
# Fast-reply defaults: MINIMAL/EASY, vad 120, auto-detect loopback, model chain
# ---------------------------------------------------------------------------

def test_default_config_new_fast_defaults():
    assert cm.DEFAULT_CONFIG["thinking_level"] == "MINIMAL"
    assert cm.DEFAULT_CONFIG["difficulty_level"] == "EASY"
    assert cm.DEFAULT_CONFIG["vad_threshold_db"] == 120
    assert cm.DEFAULT_CONFIG["selected_loopback_name"] == ""
    assert cm.DEFAULT_CONFIG["model"] == "gemini-3.8-live"
    assert cm.DEFAULT_CONFIG["fallback_model"] == "gemini-3.1-flash-live-preview"
    assert cm.DEFAULT_CONFIG["ultimate_fallback_model"] == "gemini-2.5-flash-native-audio-preview-12-2025"
    assert cm.DEFAULT_CONFIG["fallback_model"] != cm.DEFAULT_CONFIG["ultimate_fallback_model"]
    assert cm.DEFAULT_CONFIG["fallback_model"] != cm.DEFAULT_CONFIG["model"]
    assert cm.DEFAULT_CONFIG["ultimate_fallback_model"] != cm.DEFAULT_CONFIG["model"]


def test_validate_returns_new_defaults():
    cfg = cm.validate_config({})
    assert cfg["thinking_level"] == "MINIMAL"
    assert cfg["difficulty_level"] == "EASY"
    assert cfg["vad_threshold_db"] == 120
    assert cfg["selected_loopback_name"] == ""
    assert cfg["model"] == "gemini-3.8-live"
    assert cfg["fallback_model"] == "gemini-3.1-flash-live-preview"
    assert cfg["ultimate_fallback_model"] == "gemini-2.5-flash-native-audio-preview-12-2025"


def test_vad_120_within_clamp_and_auto_detect_loopback():
    # 120 must survive validation (clamp is 50..2000)
    assert cm.validate_config({"vad_threshold_db": 120})["vad_threshold_db"] == 120
    assert cm.validate_config({"vad_threshold_db": 99999})["vad_threshold_db"] == 2000
    assert cm.validate_config({"vad_threshold_db": 1})["vad_threshold_db"] == 50
    # pinned speaker default is gone: auto-detect
    assert "Speakers (Realtek" not in str(cm.DEFAULT_CONFIG.values())
    assert cm.validate_config({})["selected_loopback_name"] == ""


def test_no_1008_in_config():
    assert "1008" not in repr(cm.DEFAULT_CONFIG)
    for key, value in cm.DEFAULT_CONFIG.items():
        assert "1008" not in str(key)
        assert "1008" not in str(value)
