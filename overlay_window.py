"""Stealth HUD overlay: capture-excluded, Alt+Tab-hidden, modern dark UI (PySide6)."""

from __future__ import annotations

import html
import logging
import os
import re
import sys
from typing import Optional

from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtGui import QFont, QTextCursor

log = logging.getLogger("InterviewCopilot.overlay")

WDA_EXCLUDEFROMCAPTURE = 0x00000011
WDA_NONE = 0x00000000


def _windows_build() -> int:
    try:
        if sys.platform != "win32":
            return 0
        import platform
        ver = str(platform.version() or "")
        # "10.0.19041" -> build 19041
        parts = ver.split(".")
        return int(parts[-1]) if parts else 0
    except Exception:
        return 0


def is_capture_exclusion_supported() -> bool:
    """WDA_EXCLUDEFROMCAPTURE needs Win10 2004+ (build 19041+)."""
    try:
        if sys.platform != "win32":
            return False
        return _windows_build() >= 19041
    except Exception:
        return False


def apply_capture_exclusion(widget: QWidget) -> bool:
    """Hide window from screenshots / screenshare. Returns True on success."""
    try:
        import ctypes
        hwnd = int(widget.winId())
        res = ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
        if not res:
            log.warning("SetWindowDisplayAffinity failed (build=%s).", _windows_build())
            return False
        return True
    except Exception as e:
        log.debug("SetWindowDisplayAffinity failed: %s", e)
        return False


def get_capture_exclusion_status(widget: QWidget) -> bool:
    """Verify affinity is actually EXCLUDEFROMCAPTURE."""
    try:
        import ctypes
        hwnd = int(widget.winId())
        aff = ctypes.c_uint32(0)
        ok = ctypes.windll.user32.GetWindowDisplayAffinity(hwnd, ctypes.byref(aff))
        return bool(ok) and int(aff.value) == WDA_EXCLUDEFROMCAPTURE
    except Exception:
        return False


def _render_markdown_lite(text: str) -> str:
    """Minimal markdown -> HTML: code blocks, inline code, bullets, bold, headings."""
    try:
        safe = html.escape(text)
        # code blocks ```...```
        parts = re.split(r"```", safe)
        out: list[str] = []
        for i, chunk in enumerate(parts):
            if i % 2 == 1:
                out.append(
                    f'<pre style="background:rgba(255,255,255,8);border:1px solid rgba(255,255,255,14);'
                    f'border-radius:8px;padding:8px;font-family:Consolas,monospace;font-size:12px;'
                    f'color:#aef3c1;white-space:pre-wrap;">{chunk.strip(chr(10))}</pre>'
                )
            else:
                c = chunk
                # inline code `x`
                c = re.sub(r"`([^`]+)`", r'<code style="background:rgba(255,255,255,10);padding:1px 5px;border-radius:4px;color:#ffd479;">\1</code>', c)
                # bold **x**
                c = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", c)
                # headings ### x
                c = re.sub(r"(?m)^#{1,4}\s*(.+)$", r'<div style="color:#7ef0c1;font-weight:bold;margin:6px 0 2px;">\1</div>', c)
                # bullets - / * / •
                c = re.sub(r"(?m)^\s*[-*•]\s+(.+)$", r'<div style="margin:2px 0;">• \1</div>', c)
                # numbered 1. x
                c = re.sub(r"(?m)^\s*(\d+)[.)]\s+(.+)$", r'<div style="margin:2px 0;"><b style="color:#7ef0c1;">\1.</b> \2</div>', c)
                c = c.replace("\n", "<br>")
                out.append(f'<span style="color:#f2f2f2;">{c}</span>')
        return "".join(out)
    except Exception:
        return html.escape(text).replace("\n", "<br>")


class OverlayWindow(QWidget):
    signal_pause_toggle = Signal()
    signal_clear = Signal()
    signal_quit = Signal()
    signal_hide_toggle = Signal()
    signal_clickthrough_toggle = Signal()
    # thread-safe bridges: emit from asyncio/audio threads, delivered via event loop
    text_signal = Signal(str)
    status_signal = Signal(str)
    opacity_signal = Signal(float)

    def __init__(
        self,
        font_size: int = 14,
        opacity: float = 0.85,
        auto_resize: bool = True,
        parent=None,
    ) -> None:
        super().__init__(
            parent,
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool,
        )
        # Stealth: Tool => no taskbar button, no Alt+Tab / Win+Tab entry.
        # Frameless+TopMost => pinned HUD. Capture exclusion applied in showEvent.
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setWindowTitle("System Audio Helper")
        self.resize(460, 340)
        self._click_through = False
        self._paused = False
        self._drag_pos = None
        self._hotkeys = None
        self._auto_resize = bool(auto_resize)
        self._stealth_ok: Optional[bool] = None

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        self.container = QWidget(self)
        self.container.setObjectName("hudContainer")
        self.container.setStyleSheet(
            "#hudContainer { background: rgba(16, 20, 24, 228); border-radius: 14px; "
            "border: 1px solid rgba(46, 204, 113, 45); }"
        )
        root.addWidget(self.container)
        lay = QVBoxLayout(self.container)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)

        # header (drag handle)
        header = QHBoxLayout()
        header.setSpacing(6)
        self.status_dot = QLabel("●", self.container)
        self.status_dot.setStyleSheet("color: #e33; font-size: 15px;")
        self.status_dot.setToolTip("Disconnected")
        header.addWidget(self.status_dot)

        self.title_label = QLabel("Live hints • system audio", self.container)
        self.title_label.setStyleSheet("color: #9fb3a8; font-size: 11px; font-weight: 600;")
        header.addWidget(self.title_label, 1)

        self.stealth_pill = QLabel("STEALTH ?", self.container)
        self.stealth_pill.setStyleSheet(
            "color: #f39c12; font-size: 10px; font-weight: bold; background: rgba(243,156,18,18);"
            "border: 1px solid rgba(243,156,18,40); border-radius: 8px; padding: 2px 8px;"
        )
        self.stealth_pill.setToolTip("Capture-exclusion status (screenshot / screenshare protection).")
        header.addWidget(self.stealth_pill)

        for text, tip, slot in (
            ("⏸", "Pause/Resume (Ctrl+Alt+P)", self.signal_pause_toggle.emit),
            ("🗑", "Clear (Ctrl+Alt+C)", self._on_clear_btn),
            ("⧉", "Click-through (Ctrl+Alt+T)", self.signal_clickthrough_toggle.emit),
            ("—", "Hide (Ctrl+Alt+H)", self.hide),
            ("✕", "Quit (Ctrl+Alt+Q)", self._on_quit_btn),
        ):
            b = QPushButton(text, self.container)
            b.setFixedSize(30, 24)
            b.setToolTip(tip)
            b.setStyleSheet(
                "QPushButton { background: rgba(255,255,255,10); color: #ddd; border: none; "
                "border-radius: 6px; font-size: 12px; }"
                "QPushButton:hover { background: rgba(46,204,113,35); color: #fff; }"
            )
            b.setFocusPolicy(Qt.NoFocus)
            b.clicked.connect(slot)
            header.addWidget(b)
        lay.addLayout(header)

        # divider
        div = QLabel("", self.container)
        div.setFixedHeight(1)
        div.setStyleSheet("background: rgba(255,255,255,10); border-radius: 1px;")
        lay.addWidget(div)

        self.viewer = QTextBrowser(self.container)
        self.viewer.setOpenExternalLinks(False)
        self.viewer.setStyleSheet(
            "QTextBrowser { background: transparent; border: none; color: #f2f2f2; "
            "selection-background-color: rgba(46,204,113,60); }"
            "QScrollBar:vertical { background: transparent; width: 8px; }"
            "QScrollBar::handle:vertical { background: rgba(255,255,255,25); border-radius: 4px; min-height: 30px; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"
        )
        self.viewer.setPlaceholderText("Waiting for system audio… play a meeting / video to begin.")
        lay.addWidget(self.viewer, 1)

        bottom = QHBoxLayout()
        self.hint_label = QLabel(
            "Ctrl+Alt+H hide • C clear • P pause • T click-through • Q quit",
            self.container,
        )
        self.hint_label.setStyleSheet("color: #5f6f66; font-size: 10px;")
        bottom.addWidget(self.hint_label, 1)
        self.copy_btn = QPushButton("Copy", self.container)
        self.copy_btn.setFixedSize(58, 22)
        self.copy_btn.setToolTip("Copy all text to clipboard")
        self.copy_btn.setStyleSheet(
            "QPushButton { background: rgba(255,255,255,8); color: #9fb3a8; border: 1px solid rgba(255,255,255,12);"
            "border-radius: 6px; font-size: 10px; }"
            "QPushButton:hover { background: rgba(46,204,113,30); color: #fff; }"
        )
        self.copy_btn.setFocusPolicy(Qt.NoFocus)
        self.copy_btn.clicked.connect(self._on_copy)
        bottom.addWidget(self.copy_btn)
        lay.addLayout(bottom)

        # wire internal signals (also triggered by global hotkeys from other threads)
        self.signal_clear.connect(self.clear_text)
        self.signal_hide_toggle.connect(self._toggle_visibility)
        self.signal_clickthrough_toggle.connect(self._toggle_clickthrough)
        # thread-safe bridges from asyncio thread (QueuedConnection default)
        try:
            self.text_signal.connect(self.append_text_delta, Qt.QueuedConnection)
        except Exception:
            self.text_signal.connect(self.append_text_delta)
        try:
            self.status_signal.connect(self.set_status, Qt.QueuedConnection)
        except Exception:
            self.status_signal.connect(self.set_status)
        try:
            self.opacity_signal.connect(self.set_hud_opacity, Qt.QueuedConnection)
        except Exception:
            self.opacity_signal.connect(self.set_hud_opacity)

        self.set_font_size(font_size)
        self.set_hud_opacity(opacity)
        self.set_status("disconnected")
        self._update_stealth_pill(None)

    # -- stealth ------------------------------------------------------
    def _ensure_exclusion(self) -> bool:
        ok = apply_capture_exclusion(self)
        try:
            verified = get_capture_exclusion_status(self)
        except Exception:
            verified = ok
        final = bool(ok and verified) if sys.platform == "win32" else bool(ok)
        # On non-Windows there is no affinity API — report unknown.
        if sys.platform != "win32":
            self._update_stealth_pill(None)
            return False
        self._update_stealth_pill(final)
        if not final:
            log.warning("Capture exclusion NOT active (screenshots may include HUD).")
        return final

    def _update_stealth_pill(self, ok: Optional[bool]) -> None:
        self._stealth_ok = ok
        try:
            if ok is True:
                self.stealth_pill.setText("STEALTH ON")
                self.stealth_pill.setStyleSheet(
                    "color: #2ecc71; font-size: 10px; font-weight: bold; background: rgba(46,204,113,18);"
                    "border: 1px solid rgba(46,204,113,45); border-radius: 8px; padding: 2px 8px;"
                )
                self.stealth_pill.setToolTip("Hidden from screenshots & screenshare (WDA_EXCLUDEFROMCAPTURE). Not in Alt+Tab.")
            elif ok is False:
                self.stealth_pill.setText("STEALTH OFF")
                self.stealth_pill.setStyleSheet(
                    "color: #ff6b6b; font-size: 10px; font-weight: bold; background: rgba(255,80,80,15);"
                    "border: 1px solid rgba(255,80,80,45); border-radius: 8px; padding: 2px 8px;"
                )
                self.stealth_pill.setToolTip("Capture exclusion FAILED — HUD may appear in screenshots. Requires Win10 2004+.")
            else:
                self.stealth_pill.setText("STEALTH ?")
                self.stealth_pill.setStyleSheet(
                    "color: #f39c12; font-size: 10px; font-weight: bold; background: rgba(243,156,18,18);"
                    "border: 1px solid rgba(243,156,18,40); border-radius: 8px; padding: 2px 8px;"
                )
                self.stealth_pill.setToolTip("Capture-exclusion status unknown (non-Windows or not yet verified).")
        except Exception:
            pass

    def is_stealth_active(self) -> bool:
        return self._stealth_ok is True

    # -- public API ---------------------------------------------------
    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        # affinity must be applied after native handle exists; retry a few times
        QTimer.singleShot(100, self._ensure_exclusion)
        QTimer.singleShot(800, self._ensure_exclusion)

    def changeEvent(self, event) -> None:  # type: ignore[override]
        try:
            super().changeEvent(event)
        except Exception:
            pass
        try:
            QTimer.singleShot(150, self._ensure_exclusion)
        except Exception:
            pass

    @Slot(str)
    def append_text_delta(self, text: str) -> None:
        try:
            rendered = _render_markdown_lite(text)
            sb = self.viewer.verticalScrollBar()
            at_bottom = sb.value() >= sb.maximum() - 4
            self.viewer.insertHtml(rendered)
            # keep cursor at end so subsequent inserts append
            try:
                cursor = self.viewer.textCursor()
                cursor.movePosition(QTextCursor.MoveMode.End)
                self.viewer.setTextCursor(cursor)
            except Exception:
                pass
            if at_bottom:
                sb.setValue(sb.maximum())
            self._maybe_autoresize()
        except Exception as e:
            log.debug("append failed: %s", e)

    def append_status_line(self, text: str) -> None:
        try:
            safe = html.escape(text)
            sb = self.viewer.verticalScrollBar()
            at_bottom = sb.value() >= sb.maximum() - 4
            self.viewer.insertHtml(
                f'<div style="color:#7ef0c1;font-size:11px;margin:4px 0;">{safe}</div>'
            )
            if at_bottom:
                sb.setValue(sb.maximum())
        except Exception:
            pass

    def clear_text(self) -> None:
        try:
            self.viewer.clear()
            if self._auto_resize:
                self.resize(460, 340)
        except Exception:
            pass

    @Slot(str)
    def set_status(self, state: str) -> None:
        colors = {"active": ("#2ecc71", "Active — streaming system audio"),
                  "paused": ("#f39c12", "Paused — context saver on"),
                  "disconnected": ("#e33", "Disconnected"),
                  "connecting": ("#f39c12", "Connecting…")}
        color, tip = colors.get(state, colors["disconnected"])
        try:
            self.status_dot.setStyleSheet(f"color: {color}; font-size: 15px;")
            self.status_dot.setToolTip(tip)
            self.title_label.setText(f"Live hints • {tip}")
            self._paused = (state == "paused")
            try:
                glow = {"active": "rgba(46,204,113,55)", "paused": "rgba(243,156,18,55)",
                        "connecting": "rgba(243,156,18,55)"}.get(state, "rgba(255,255,255,18)")
                self.container.setStyleSheet(
                    f"#hudContainer {{ background: rgba(16, 20, 24, 228); border-radius: 14px; "
                    f"border: 1px solid {glow}; }}"
                )
            except Exception:
                pass
        except Exception:
            pass

    def set_font_size(self, size: int) -> None:
        try:
            size = max(8, min(28, int(size)))
            f = self.viewer.font()
            if isinstance(f, QFont):
                f.setPointSize(size)
                self.viewer.setFont(f)
            else:
                font = QFont()
                font.setPointSize(size)
                self.viewer.setFont(font)
        except Exception:
            pass

    @Slot(float)
    def set_hud_opacity(self, value: float) -> None:
        try:
            self.setWindowOpacity(max(0.3, min(1.0, float(value))))
        except Exception:
            pass

    def set_auto_resize(self, enabled: bool) -> None:
        self._auto_resize = bool(enabled)
        if not self._auto_resize:
            self.resize(460, 340)

    def is_auto_resize(self) -> bool:
        return self._auto_resize

    def _maybe_autoresize(self) -> None:
        """Grow window height with content up to 60% of screen height."""
        if not self._auto_resize:
            return
        try:
            doc_h = float(self.viewer.document().size().height())
            # header + hint + margins + padding
            extra = 150
            new_h = int(doc_h + extra)
            new_h = max(340, new_h)
            try:
                screen = self.screen()
                avail = screen.availableGeometry() if screen is not None else None
                if avail is not None:
                    cap = int(avail.height() * 0.6)
                else:
                    cap = 700
            except Exception:
                cap = 700
            new_h = min(new_h, max(340, cap))
            if abs(new_h - self.height()) >= 8:
                self.resize(max(460, self.width()), new_h)
        except Exception as e:
            log.debug("autoresize failed: %s", e)

    def set_click_through(self, enabled: bool) -> None:
        try:
            self._click_through = bool(enabled)
            self.setWindowFlag(Qt.WindowTransparentForInput, self._click_through)
            self.show()  # flag change requires re-show
            QTimer.singleShot(100, self._ensure_exclusion)
        except Exception as e:
            log.debug("click-through toggle failed: %s", e)

    def is_click_through(self) -> bool:
        return self._click_through

    # -- drag ---------------------------------------------------------
    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        try:
            if event.button() == Qt.LeftButton:
                self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
        except Exception:
            self._drag_pos = None

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        try:
            if self._drag_pos is not None and event.buttons() & Qt.LeftButton:
                self.move(event.globalPosition().toPoint() - self._drag_pos)
        except Exception:
            pass

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        self._drag_pos = None

    def closeEvent(self, event) -> None:  # type: ignore[override]
        """Closing the HUD kills the entire program (no orphan audio/network threads)."""
        try:
            self.stop_hotkeys()
        except Exception:
            pass
        try:
            app = QApplication.instance()
            if app is not None:
                app.quit()
        except Exception:
            pass
        try:
            super().closeEvent(event)
        except Exception:
            pass
        finally:
            os._exit(0)

    # -- internal slots ------------------------------------------------
    def _toggle_visibility(self) -> None:
        try:
            if self.isVisible():
                self.hide()
            else:
                self.show()
                QTimer.singleShot(120, self._ensure_exclusion)
        except Exception:
            pass

    def _toggle_clickthrough(self) -> None:
        try:
            self.set_click_through(not self._click_through)
        except Exception:
            pass

    def _on_clear_btn(self) -> None:
        self.clear_text()
        self.signal_clear.emit()

    def _on_copy(self) -> None:
        try:
            cb = QApplication.clipboard()
            if cb is not None:
                cb.setText(self.viewer.toPlainText())
                self.copy_btn.setText("Copied!")
                QTimer.singleShot(1200, lambda: self.copy_btn.setText("Copy"))
        except Exception:
            pass

    def _on_quit_btn(self) -> None:
        self.signal_quit.emit()

    # -- global hotkeys ------------------------------------------------
    def start_hotkeys(self) -> None:
        try:
            from pynput import keyboard
        except Exception as e:
            log.warning("pynput unavailable, hotkeys disabled: %s", e)
            return
        if self._hotkeys is not None:
            return

        def _do_hide_show():
            try:
                self.signal_hide_toggle.emit()
            except Exception:
                pass

        def _do_clear():
            try:
                self.signal_clear.emit()
            except Exception:
                pass

        def _do_pause():
            try:
                self.signal_pause_toggle.emit()
            except Exception:
                pass

        def _do_click():
            try:
                self.signal_clickthrough_toggle.emit()
            except Exception:
                pass

        def _do_quit():
            try:
                self.signal_quit.emit()
            finally:
                os._exit(0)

        try:
            self._hotkeys = keyboard.GlobalHotKeys({
                "<ctrl>+<alt>+h": _do_hide_show,
                "<ctrl>+<alt>+c": _do_clear,
                "<ctrl>+<alt>+p": _do_pause,
                "<ctrl>+<alt>+t": _do_click,
                "<ctrl>+<alt>+q": _do_quit,
            })
            self._hotkeys.start()
            log.info("Global hotkeys registered.")
        except Exception as e:
            log.warning("Hotkey registration failed: %s", e)
            self._hotkeys = None

    def stop_hotkeys(self) -> None:
        try:
            if self._hotkeys is not None:
                self._hotkeys.stop()
        except Exception:
            pass
        self._hotkeys = None
