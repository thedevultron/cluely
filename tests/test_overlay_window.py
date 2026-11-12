"""Offscreen Qt tests for OverlayWindow (_render_markdown_lite, HUD)."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

# Make repo root importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    # no quit here to avoid killing pytest session


@pytest.fixture
def hud(qapp):
    from overlay_window import OverlayWindow

    h = OverlayWindow(font_size=14, opacity=0.85, auto_resize=True)
    h.show()
    qapp.processEvents()
    yield h
    try:
        h.stop_hotkeys()
    except Exception:
        pass
    h.close()
    # ensure offscreen close does not trigger os._exit path
    h.deleteLater()
    qapp.processEvents()


# ---------------------------------------------------------------- markdown


def test_markdown_escapes_html_no_xss():
    from overlay_window import _render_markdown_lite

    out = _render_markdown_lite('<script>alert("x")</script> hello')
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_markdown_escapes_img_onerror():
    from overlay_window import _render_markdown_lite

    out = _render_markdown_lite('<img src=x onerror=alert(1)>')
    assert "<img" not in out
    assert "&lt;img" in out


def test_markdown_bold_code_bullets_headings():
    from overlay_window import _render_markdown_lite

    out = _render_markdown_lite("### Title\n- one\n- two\n**bold** and `code`")
    assert "<b>bold</b>" in out
    assert "<code" in out and "code" in out
    # bullet marker rendered as • inside div
    assert "•" in out
    assert "Title" in out
    assert "font-weight" in out  # heading styling


def test_markdown_no_double_br():
    from overlay_window import _render_markdown_lite

    out = _render_markdown_lite("### Hi\nline1\nline2")
    # heading div should not be followed by stray double <br><br> from \n replacement
    assert "<br><br><br>" not in out
    # and not <div...>...<br> double after heading
    # count consecutive br: should be at most 1 between plain lines
    assert "<br><br><br>" not in out
    assert "Hi" in out
    assert "line1" in out
    assert "line2" in out


def test_markdown_code_block():
    from overlay_window import _render_markdown_lite

    out = _render_markdown_lite("```\nprint('<hi>')\n```")
    assert "<pre" in out
    assert "&lt;hi&gt;" in out
    assert "<hi>" not in out


def test_markdown_inline_code_not_bold_processed():
    from overlay_window import _render_markdown_lite

    out = _render_markdown_lite("`**not bold**`")
    # inside code, ** should remain literal
    assert "**not bold**" in out
    assert "<b>not bold" not in out


def test_markdown_numbered_list():
    from overlay_window import _render_markdown_lite

    out = _render_markdown_lite("1. first")
    assert "first" in out


# ---------------------------------------------------------------- signals & slots


def test_text_signal_emits_and_appends(hud, qapp):
    got = []

    def _spy(text):
        got.append(text)

    hud.append_text_delta("hello world")
    qapp.processEvents()
    assert "hello world" in hud.viewer.toPlainText()

    # thread-safe bridge: signal -> slot
    hud.text_signal.emit("via signal")
    qapp.processEvents()
    assert "via signal" in hud.viewer.toPlainText()


def test_status_signal_and_colors(hud, qapp):
    hud.set_status("active")
    assert "#27f795" in hud.status_dot.styleSheet()
    hud.set_status("paused")
    assert "#ffb454" in hud.status_dot.styleSheet()
    assert hud._paused is True
    hud.set_status("disconnected")
    assert "#e33" in hud.status_dot.styleSheet() or "#ff" in hud.status_dot.styleSheet()
    assert hud._paused is False

    hud.status_signal.emit("active")
    qapp.processEvents()
    assert "#27f795" in hud.status_dot.styleSheet()


def test_opacity_signal_and_clamping(hud, qapp):
    hud.set_hud_opacity(2.5)
    assert hud.windowOpacity() == pytest.approx(1.0, abs=0.02)
    hud.set_hud_opacity(0.05)
    assert hud.windowOpacity() == pytest.approx(0.3, abs=0.02)
    hud.opacity_signal.emit(0.5)
    qapp.processEvents()
    assert hud.windowOpacity() == pytest.approx(0.5, abs=0.02)


def test_font_size_clamping(hud):
    hud.set_font_size(100)
    assert hud.viewer.font().pointSize() == 28
    hud.set_font_size(1)
    assert hud.viewer.font().pointSize() == 8
    hud.set_font_size(16)
    assert hud.viewer.font().pointSize() == 16


def test_clear_text_slot(hud, qapp):
    hud.append_text_delta("to be cleared")
    assert "to be cleared" in hud.viewer.toPlainText()
    hud.signal_clear.emit()
    qapp.processEvents()
    assert hud.viewer.toPlainText().strip() == ""


def test_append_status_line_escapes(hud):
    hud.append_status_line("<b>not html</b>")
    assert "&lt;b&gt;" in hud.viewer.document().toHtml() or "not html" in hud.viewer.toPlainText()
    # raw HTML tag should not be interpreted as formatting block div directly from user input
    assert hud.viewer.toPlainText().count("not html") >= 1


def test_auto_trim_caps_history(hud, qapp):
    max_chars = getattr(hud, "MAX_HISTORY_CHARS", 20000)
    # append many chunks exceeding cap
    chunk = "x" * 5000 + "\n"
    for _ in range(8):
        hud.append_text_delta(chunk)
    qapp.processEvents()
    assert hud.viewer.document().characterCount() <= max_chars + 2000


def test_click_through_toggle(hud, qapp):
    assert hud.is_click_through() is False
    hud.signal_clickthrough_toggle.emit()
    qapp.processEvents()
    assert hud.is_click_through() is True
    hud.set_click_through(False)
    assert hud.is_click_through() is False


def test_pause_hide_quit_signals_do_not_kill(hud, qapp):
    # signals should emit without os._exit
    received = []
    hud.signal_pause_toggle.connect(lambda: received.append("pause"))
    hud.signal_quit.connect(lambda: received.append("quit"))
    hud.signal_hide_toggle.connect(lambda: received.append("hide"))
    hud.signal_pause_toggle.emit()
    hud.signal_quit.emit()
    hud.signal_hide_toggle.emit()
    qapp.processEvents()
    assert received == ["pause", "quit", "hide"]


def test_stealth_window_flags(hud):
    flags = hud.windowFlags()
    assert flags & Qt.FramelessWindowHint
    assert flags & Qt.WindowStaysOnTopHint
    assert flags & Qt.Tool  # no Alt+Tab / taskbar


def test_stealth_pill_states(hud):
    hud._update_stealth_pill(True)
    assert "ON" in hud.stealth_pill.text()
    hud._update_stealth_pill(False)
    assert "OFF" in hud.stealth_pill.text()
    hud._update_stealth_pill(None)
    assert "?" in hud.stealth_pill.text()
    assert hud.is_stealth_active() is False
    hud._update_stealth_pill(True)
    assert hud.is_stealth_active() is True


def test_auto_resize_toggle(hud):
    hud.set_auto_resize(False)
    assert hud.is_auto_resize() is False
    hud.set_auto_resize(True)
    assert hud.is_auto_resize() is True


def test_start_stop_hotkeys_idempotent(hud):
    # should not raise even if pynput missing
    hud.start_hotkeys()
    hud.start_hotkeys()  # second call no-op / no exception
    hud.stop_hotkeys()
    hud.stop_hotkeys()
    assert hud._hotkeys is None
