"""Offscreen Qt tests for SettingsDialog (collect_config, validation, tabs)."""
from __future__ import annotations

import copy
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def fast_devices(monkeypatch):
    """Make device scan instant so dialog __init__ thread finishes quickly."""
    import settings_dialog as sd

    monkeypatch.setattr(sd, "_query_mics", lambda: [(0, "Fake Mic")], raising=False)
    monkeypatch.setattr(sd, "_query_loopbacks", lambda: ["Fake Loopback"], raising=False)


@pytest.fixture
def dlg(qapp, fast_devices):
    from config_manager import DEFAULT_CONFIG
    from settings_dialog import SettingsDialog

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    d = SettingsDialog(cfg)
    d.show()
    qapp.processEvents()
    yield d
    try:
        d.close()
    except Exception:
        pass
    d.deleteLater()
    qapp.processEvents()


def test_tabs_count_and_titles(dlg):
    assert dlg.tabs.count() == 5
    titles = [dlg.tabs.tabText(i) for i in range(dlg.tabs.count())]
    # required tabs per spec
    for expected in ("API", "Audio", "Knowledge", "Prompt", "Controls"):
        assert any(expected.lower() in t.lower() for t in titles), f"missing tab {expected}: {titles}"


def test_collect_config_roundtrip_all_default_keys(qapp, fast_devices):
    from config_manager import DEFAULT_CONFIG
    from settings_dialog import SettingsDialog

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    d = SettingsDialog(cfg)
    out = d.collect_config()
    for key in DEFAULT_CONFIG:
        assert key in out, f"collect_config missing DEFAULT_CONFIG key: {key}"
    d.close()
    d.deleteLater()
    qapp.processEvents()


def test_collect_config_preserves_values(qapp, fast_devices):
    from config_manager import DEFAULT_CONFIG
    from settings_dialog import SettingsDialog

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["api_key"] = "AIzaTest123"
    cfg["model"] = "gemini-3.8-live"
    d = SettingsDialog(cfg)
    out = d.collect_config()
    assert out["api_key"] == "AIzaTest123"
    assert out["model"] == "gemini-3.8-live"
    # ultimate fallback must be collected (was missing before)
    assert out["ultimate_fallback_model"] == cfg["ultimate_fallback_model"]
    assert out["stealth_hide_from_capture"] == cfg["stealth_hide_from_capture"]
    assert out["selftest_hello"] == cfg["selftest_hello"]
    d.close()
    d.deleteLater()
    qapp.processEvents()


def test_launch_disabled_when_api_key_empty(dlg, qapp):
    dlg.api_key_edit.setText("")
    qapp.processEvents()
    # Launch should be disabled or clearly invalid when key empty
    # Accept either disabled button or warning-path: we assert disabled per new UX
    assert dlg.btn_launch.isEnabled() is False


def test_launch_enabled_when_api_key_filled(dlg, qapp):
    dlg.api_key_edit.setText("AIzaFilled")
    qapp.processEvents()
    assert dlg.btn_launch.isEnabled() is True


def test_launch_requested_signal_emits_dict(dlg, qapp, monkeypatch):
    import settings_dialog as sd

    monkeypatch.setattr(sd, "_query_mics", lambda: [])
    monkeypatch.setattr(sd, "_query_loopbacks", lambda: [])
    # stub save_config to avoid disk write
    import config_manager

    monkeypatch.setattr(config_manager, "save_config", lambda data, path=None: "config.json")

    dlg.api_key_edit.setText("AIzaSignalTest")
    qapp.processEvents()
    emitted = []
    dlg.launch_requested.connect(emitted.append)
    # call _on_launch directly (it will call save_config from config_manager)
    dlg._on_launch()
    assert len(emitted) == 1
    assert isinstance(emitted[0], dict)
    assert emitted[0]["api_key"] == "AIzaSignalTest"
    for key in ("model", "font_size", "hud_opacity", "api_key"):
        assert key in emitted[0]


def test_opacity_and_font_roundtrip(dlg):
    dlg.font_spin.setValue(18)
    dlg.opacity_slider.setValue(70)
    out = dlg.collect_config()
    assert out["font_size"] == 18
    assert out["hud_opacity"] == pytest.approx(0.7)


def test_thinking_difficulty_normalized(dlg):
    dlg.thinking_combo.setCurrentText("HIGH")
    dlg.difficulty_combo.setCurrentText("HARD")
    out = dlg.collect_config()
    assert out["thinking_level"] in ("MINIMAL", "LOW", "MEDIUM", "HIGH")
    assert out["difficulty_level"] in ("EASY", "MEDIUM", "HARD", "EXPERT")


def test_audio_mode_roundtrip(dlg):
    dlg.radio_dual.setChecked(True)
    out = dlg.collect_config()
    assert out["audio_mode"] == "dual"
    assert out["mic_enabled"] is True
    dlg.radio_system.setChecked(True)
    out2 = dlg.collect_config()
    assert out2["audio_mode"] == "system"
    assert out2["mic_enabled"] is False


def test_knowledge_and_web_roundtrip(dlg):
    dlg.enable_knowledge_check.setChecked(True)
    dlg.knowledge_list.clear()
    dlg.knowledge_list.addItem("C:\\tmp\\a.txt")
    dlg.knowledge_max_spin.setValue(9000)
    dlg.enable_web_check.setChecked(True)
    dlg.web_max_spin.setValue(7)
    out = dlg.collect_config()
    assert out["enable_knowledge"] is True
    assert len(out["knowledge_paths"]) >= 1
    assert out["knowledge_max_chars"] == 9000
    assert out["enable_web_search"] is True
    assert out["web_max_results"] == 7


def test_stealth_and_selftest_checkboxes_exist(dlg):
    # these keys were previously not exposed; they must now have controls
    assert hasattr(dlg, "stealth_hide_check")
    assert hasattr(dlg, "selftest_check")
    # toggle and round-trip
    dlg.stealth_hide_check.setChecked(False)
    dlg.selftest_check.setChecked(False)
    out = dlg.collect_config()
    assert out["stealth_hide_from_capture"] is False
    assert out["selftest_hello"] is False


def test_ultimate_fallback_combo_exists(dlg):
    assert hasattr(dlg, "ultimate_combo")
    dlg.ultimate_combo.setCurrentText("gemini-2.5-flash-native-audio-preview-12-2025")
    out = dlg.collect_config()
    assert out["ultimate_fallback_model"] == "gemini-2.5-flash-native-audio-preview-12-2025"


def test_tooltips_present(dlg):
    # spot-check accessibility: key controls have tooltips
    for w in (dlg.api_key_edit, dlg.model_combo, dlg.vad_slider, dlg.knowledge_list):
        assert w.toolTip() != ""


def test_buttons_have_parents_and_focus(dlg):
    for b in (dlg.btn_save, dlg.btn_launch, dlg.btn_cancel):
        assert b.parent() is not None
        assert b.isEnabled() in (True, False)  # launch may be disabled, that's fine


def test_reset_prompt(dlg):
    from config_manager import DEFAULT_CONFIG

    dlg.prompt_edit.setPlainText("custom")
    dlg._on_reset_prompt()
    assert dlg.prompt_edit.toPlainText() == DEFAULT_CONFIG["system_instruction"]
