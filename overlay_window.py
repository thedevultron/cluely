"""Stealth HUD overlay: capture-excluded, Alt+Tab-hidden, modern dark UI (PySide6)."""
from __future__ import annotations

import html
import logging
import re
import sys
from typing import Optional

from PySide6.QtCore import Qt, Signal
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
    if sys.platform != "win32":
        return False
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
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        hwnd = int(widget.winId())
        aff = ctypes.c_uint32(0)
        ok = ctypes.windll.user32.GetWindowDisplayAffinity(hwnd, ctypes.byref(aff))
        return bool(ok) and int(aff.value) == WDA_EXCLUDEFROMCAPTURE
    except Exception:
        return False


# ------------------------------------------------------------------ markdown
# Escape first, then apply a small safe allow-list of replacements.
# The user input can never inject HTML: every "<", ">", "&" is escaped before
# we add our own tags. Inline code is protected so ** inside `...` stays literal.

_CODE_FENCE_RE = re.compile(r"```([\s\S]*?)```")
_INLINE_CODE_RE = re.compile(r"`([^`]+?)`")
_BOLD_RE = re.compile(r"\*\*([^*]+?)\*\*")
_HEADING_RE = re.compile(r"^\s*#{1,4}\s+(.+)$")
_BULLET_RE = re.compile(r"^\s*[-*•]\s+(.+)$")
_NUMBERED_RE = re.compile(r"^\s*(\d+)[.)]\s+(.+)$")


def _inline(s: str) -> str:
    """Bold + inline code for a single already-escaped line/segment."""
    try:
        parts = _INLINE_CODE_RE.split(s)
        out: list[str] = []
        for i, p in enumerate(parts):
            if i % 2 == 1:
                out.append(
                    f'<code style="background:#1f1f1f;padding:1px 5px;border-radius:4px;'
                    f'font-family:"Roboto Mono","Segoe UI",Consolas,monospace;color:#27f795;">{p}</code>'
                )
            else:
                out.append(_BOLD_RE.sub(r"<b>\1</b>", p))
        return "".join(out)
    except Exception:
        return s


def _render_plain_segment(seg: str) -> str:
    """Render one non-code segment line-by-line; no double <br> after blocks."""
    try:
        lines = seg.split("\n")
        out: list[str] = []
        for idx, line in enumerate(lines):
            is_last = idx == len(lines) - 1
            if line.strip() == "":
                if not is_last:
                    out.append("<br>")
                continue
            m = _HEADING_RE.match(line)
            if m:
                out.append(
                    f'<div style="color:#27f795;font-weight:bold;margin:6px 0 2px;font-size:13px;">'
                    f"{_inline(m.group(1))}</div>"
                )
                continue
            m = _BULLET_RE.match(line)
            if m:
                out.append(f'<div style="margin:2px 0 2px 6px;">• {_inline(m.group(1))}</div>')
                continue
            m = _NUMBERED_RE.match(line)
            if m:
                out.append(
                    f'<div style="margin:2px 0 2px 6px;">'
                    f'<b style="color:#27f795;">{m.group(1)}.</b> {_inline(m.group(2))}</div>'
                )
                continue
            # plain line: <br> separates it from the next line, never doubled
            out.append(f'<span style="color:#d9d9d9;">{_inline(line)}</span>')
            if not is_last:
                out.append("<br>")
        return "".join(out)
    except Exception:
        return html.escape(seg).replace("\n", "<br>")


def _render_markdown_lite(text: str) -> str:
    """Minimal markdown -> HTML: code blocks, inline code, bullets, bold, headings.

    Escapes first, then adds only safe tags. No double <br> bugs.
    """
    try:
        safe = html.escape(str(text))
        parts = _CODE_FENCE_RE.split(safe)
        out: list[str] = []
        for i, chunk in enumerate(parts):
            if i % 2 == 1:
                # inside ``` ... ``` (already escaped, preserve newlines in <pre>)
                inner = chunk
                # strip a single surrounding newline pair added by fences, keep indent
                if inner.startswith("\n"):
                    inner = inner[1:]
                if inner.endswith("\n"):
                    inner = inner[:-1]
                out.append(
                    f'<pre style="background:#151515;border:1px solid #353535;'
                    f'border-radius:8px;padding:8px;font-family:"Roboto Mono","Segoe UI",Consolas,monospace;font-size:12px;'
                    f'color:#27f795;white-space:pre-wrap;margin:4px 0;">{inner}</pre>'
                )
            else:
                out.append(_render_plain_segment(chunk))
        return "".join(out)
    except Exception:
        try:
            return html.escape(str(text)).replace("\n", "<br>")
        except Exception:
            return ""


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

    MAX_HISTORY_CHARS = 20000

    def __init__(
        self,
        font_size: int = 14,
        opacity: float = 0.85,
        auto_resize: bool = True,
        parent=None,
    ) -> None:
        # init flags BEFORE super().__init__ triggers changeEvent/showEvent
        self._click_through = False
        self._paused = False
        self._drag_pos = None
        self._hotkeys = None
        self._auto_resize = bool(auto_resize)
        self._stealth_ok: Optional[bool] = None
        self._excl_pending = False
        self._excl_initial_done = False
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
        self.setMinimumSize(320, 220)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        self.container = QWidget(self)
        self.container.setObjectName("hudContainer")
        self.container.setStyleSheet(
            "#hudContainer { background: rgba(10, 10, 10, 235); border-radius: 8px; "
            "border: 1px solid #353535; }"
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
        self.status_dot.setAccessibleName("Audio status")
        header.addWidget(self.status_dot)

        self.title_label = QLabel("Live hints • system audio", self.container)
        self.title_label.setStyleSheet("color: #8d8d8d; font-size: 11px; font-weight: 600; font-family: \"Roboto\", \"Segoe UI\", sans-serif;")
        self.title_label.setAccessibleName("HUD title")
        header.addWidget(self.title_label, 1)

        self.stealth_pill = QLabel("STEALTH ?", self.container)
        self.stealth_pill.setStyleSheet(
            "color: #ffb454; font-size: 10px; font-weight: bold; background: rgba(255,180,84,12);"
            "border: 1px solid #353535; border-radius: 8px; padding: 2px 8px; font-family: \"Roboto Mono\", \"Segoe UI\", Consolas, monospace;"
        )
        self.stealth_pill.setToolTip("Capture-exclusion status (screenshot / screenshare protection).")
        self.stealth_pill.setAccessibleName("Stealth status")
        header.addWidget(self.stealth_pill)

        # NOTE: each button is connected exactly once. _on_clear_btn only emits
        # signal_clear (which is wired to clear_text) — no double-clear.
        header_buttons = (
            ("⏸", "Pause/Resume (Ctrl+Alt+P)", self.signal_pause_toggle.emit),
            ("🗑", "Clear (Ctrl+Alt+C)", self._on_clear_btn),
            ("⧉", "Click-through (Ctrl+Alt+T)", self.signal_clickthrough_toggle.emit),
            ("—", "Hide (Ctrl+Alt+H)", self.signal_hide_toggle.emit),
            ("✕", "Quit (Ctrl+Alt+Q)", self._on_quit_btn),
        )
        self._header_buttons: list[QPushButton] = []
        for text, tip, slot in header_buttons:
            b = QPushButton(text, self.container)
            b.setFixedSize(30, 24)
            b.setToolTip(tip)
            b.setAccessibleName(tip)
            b.setStyleSheet(
                "QPushButton { background: #262626; color: #d9d9d9; border: 1px solid #353535; "
                "border-radius: 4px; font-size: 12px; font-family: \"Roboto\", \"Segoe UI\", sans-serif; }"
                "QPushButton:hover { background: #262626; color: #ffffff; border: 1px solid #27f795; }"
                "QPushButton:pressed { background: #151515; color: #ffffff; border: 1px solid #27f795; }"
            )
            b.setFocusPolicy(Qt.NoFocus)
            b.clicked.connect(slot)
            header.addWidget(b)
            self._header_buttons.append(b)
        lay.addLayout(header)

        # divider
        div = QLabel("", self.container)
        div.setFixedHeight(1)
        div.setStyleSheet("background: #353535; border-radius: 1px;")
        lay.addWidget(div)

        self.viewer = QTextBrowser(self.container)
        self.viewer.setOpenExternalLinks(False)
        self.viewer.setAccessibleName("Live hints viewer")
        self.viewer.setAccessibleDescription("Streaming text hints from system audio. Scroll while reading.")
        self.viewer.setFocusPolicy(Qt.StrongFocus)
        self.viewer.setStyleSheet(
            "QTextBrowser { background: transparent; border: none; color: #d9d9d9; "
            "selection-background-color: rgba(39,247,149,60); font-family: \"Roboto\", \"Segoe UI\", sans-serif; }"
            "QScrollBar:vertical { background: transparent; width: 8px; }"
            "QScrollBar::handle:vertical { background: #353535; border-radius: 4px; min-height: 30px; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"
        )
        self.viewer.setPlaceholderText("Waiting for system audio… play a meeting / video to begin.")
        lay.addWidget(self.viewer, 1)

        bottom = QHBoxLayout()
        self.hint_label = QLabel(
            "Ctrl+Alt+H hide • C clear • P pause • T click-through • Q quit",
            self.container,
        )
        self.hint_label.setStyleSheet("color: #8d8d8d; font-size: 10px; font-family: \"Roboto Mono\", \"Segoe UI\", Consolas, monospace;")
        bottom.addWidget(self.hint_label, 1)
        self.copy_btn = QPushButton("Copy", self.container)
        self.copy_btn.setFixedSize(58, 22)
        self.copy_btn.setToolTip("Copy all text to clipboard")
        self.copy_btn.setAccessibleName("Copy all text")
        self.copy_btn.setStyleSheet(
            "QPushButton { background: transparent; color: #8d8d8d; border: 1px solid #353535;"
            "border-radius: 4px; font-size: 10px; font-family: \"Roboto\", \"Segoe UI\", sans-serif; }"
            "QPushButton:hover { background: #151515; color: #d9d9d9; border: 1px solid #27f795; }"
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
        try:
            self._excl_pending = False
        except Exception:
            pass
        if sys.platform != "win32":
            try:
                self._update_stealth_pill(None)
            except Exception:
                pass
            return False
        ok = apply_capture_exclusion(self)
        try:
            verified = get_capture_exclusion_status(self)
        except Exception:
            verified = ok
        final = bool(ok and verified) if sys.platform == "win32" else bool(ok)
        try:
            self._update_stealth_pill(final)
        except Exception:
            pass
        if not final:
            log.warning("Capture exclusion NOT active (screenshots may include HUD).")
        try:
            self._excl_initial_done = True
        except Exception:
            pass
        return final

    def _schedule_exclusion(self) -> None:
        try:
            if getattr(self, "_excl_pending", False):
                return
            self._excl_pending = True
            from PySide6.QtCore import QTimer

            QTimer.singleShot(100, self._ensure_exclusion_single_shot)
        except Exception:
            try:
                self._excl_pending = False
            except Exception:
                pass

    def _ensure_exclusion_single_shot(self) -> None:
        try:
            self._ensure_exclusion()
        except Exception:
            try:
                self._excl_pending = False
            except Exception:
                pass

    def _update_stealth_pill(self, ok: Optional[bool]) -> None:
        try:
            self._stealth_ok = ok
        except Exception:
            pass
        try:
            if not hasattr(self, "stealth_pill"):
                return
            if ok is True:
                self.stealth_pill.setText("STEALTH ON")
                self.stealth_pill.setStyleSheet(
                    "color: #27f795; font-size: 10px; font-weight: bold; background: rgba(39,247,149,12);"
                    "border: 1px solid #353535; border-radius: 8px; padding: 2px 8px; font-family: \"Roboto Mono\", \"Segoe UI\", Consolas, monospace;"
                )
                self.stealth_pill.setToolTip("Hidden from screenshots & screenshare (WDA_EXCLUDEFROMCAPTURE). Not in Alt+Tab.")
            elif ok is False:
                self.stealth_pill.setText("STEALTH OFF")
                self.stealth_pill.setStyleSheet(
                    "color: #ff6b6b; font-size: 10px; font-weight: bold; background: rgba(255,80,80,15);"
                    "border: 1px solid #353535; border-radius: 8px; padding: 2px 8px; font-family: \"Roboto Mono\", \"Segoe UI\", Consolas, monospace;"
                )
                self.stealth_pill.setToolTip("Capture exclusion FAILED — HUD may appear in screenshots. Requires Win10 2004+.")
            else:
                self.stealth_pill.setText("STEALTH ?")
                self.stealth_pill.setStyleSheet(
                    "color: #ffb454; font-size: 10px; font-weight: bold; background: rgba(255,180,84,12);"
                    "border: 1px solid #353535; border-radius: 8px; padding: 2px 8px; font-family: \"Roboto Mono\", \"Segoe UI\", Consolas, monospace;"
                )
                self.stealth_pill.setToolTip("Capture-exclusion status unknown (non-Windows or not yet verified).")
        except Exception:
            pass

    def is_stealth_active(self) -> bool:
        return self._stealth_ok is True

    # -- public API ---------------------------------------------------
    def showEvent(self, event) -> None:  # type: ignore[override]
        try:
            super().showEvent(event)
        except Exception:
            pass
        self._schedule_exclusion()
        try:
            from PySide6.QtCore import QTimer

            # second retry catches late native-handle creation
            QTimer.singleShot(800, self._ensure_exclusion_single_shot)
        except Exception:
            pass

    def changeEvent(self, event) -> None:  # type: ignore[override]
        try:
            super().changeEvent(event)
        except Exception:
            pass
        self._schedule_exclusion()

    def _is_at_bottom(self) -> bool:
        try:
            sb = self.viewer.verticalScrollBar()
            return sb.value() >= sb.maximum() - 4
        except Exception:
            return True

    def _trim_history(self) -> None:
        """Cap the document at MAX_HISTORY_CHARS to prevent a memory leak."""
        try:
            doc = self.viewer.document()
            total = int(doc.characterCount())
            max_chars = int(self.MAX_HISTORY_CHARS)
            if total <= max_chars:
                return
            excess = total - max_chars
            cursor = QTextCursor(doc)
            cursor.movePosition(QTextCursor.MoveOperation.Start)
            remaining = excess
            while remaining > 0:
                step = min(remaining, 2000)
                cursor.movePosition(
                    QTextCursor.MoveOperation.NextCharacter,
                    QTextCursor.MoveMode.KeepAnchor,
                    step,
                )
                remaining -= step
            cursor.removeSelectedText()
        except Exception as e:
            try:
                log.debug("trim history failed: %s", e)
            except Exception:
                pass

    def append_text_delta(self, text: str) -> None:
        try:
            if text is None:
                return
            rendered = _render_markdown_lite(str(text))
            at_bottom = self._is_at_bottom()
            self.viewer.insertHtml(rendered)
            self._trim_history()
            # only pin cursor to the end when the user was already at the bottom;
            # otherwise preserve their reading position (no autoscroll hijack).
            if at_bottom:
                try:
                    cursor = self.viewer.textCursor()
                    try:
                        cursor.movePosition(QTextCursor.MoveOperation.End)
                    except Exception:
                        cursor.movePosition(QTextCursor.MoveMode.End)
                    self.viewer.setTextCursor(cursor)
                except Exception:
                    pass
                try:
                    sb = self.viewer.verticalScrollBar()
                    sb.setValue(sb.maximum())
                except Exception:
                    pass
            self._maybe_autoresize()
        except Exception as e:
            log.debug("append failed: %s", e)

    def append_status_line(self, text: str) -> None:
        try:
            safe = html.escape(str(text))
            at_bottom = self._is_at_bottom()
            self.viewer.insertHtml(
                f'<div style="color:#27f795;font-size:11px;margin:4px 0;">{safe}</div>'
            )
            self._trim_history()
            if at_bottom:
                try:
                    sb = self.viewer.verticalScrollBar()
                    sb.setValue(sb.maximum())
                except Exception:
                    pass
        except Exception:
            pass

    def clear_text(self) -> None:
        try:
            self.viewer.clear()
            if self._auto_resize:
                self.resize(460, 340)
        except Exception:
            pass

    def set_status(self, state: str) -> None:
        try:
            s = str(state or "disconnected").lower()
            colors = {
                "active": ("#27f795", "Active — streaming system audio"),
                "paused": ("#ffb454", "Paused — context saver on"),
                "disconnected": ("#e33", "Disconnected"),
                "connecting": ("#ffb454", "Connecting…"),
            }
            color, tip = colors.get(s, colors["disconnected"])
            try:
                self.status_dot.setStyleSheet(f"color: {color}; font-size: 15px;")
                self.status_dot.setToolTip(tip)
                self.title_label.setText(f"Live hints • {tip}")
                self._paused = (s == "paused")
                try:
                    tint = {
                        "active": "#27f795",
                        "paused": "#ffb454",
                        "connecting": "#ffb454",
                    }.get(s, "#353535")
                    self.container.setStyleSheet(
                        f"#hudContainer {{ background: rgba(10, 10, 10, 235); border-radius: 8px; "
                        f"border: 1px solid {tint}; }}"
                    )
                except Exception:
                    pass
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
            was_visible = self.isVisible()
            self.setWindowFlag(Qt.WindowTransparentForInput, self._click_through)
            # flag change requires re-show, but never un-hide a hidden HUD
            if was_visible:
                self.show()
            self._schedule_exclusion()
        except Exception as e:
            log.debug("click-through toggle failed: %s", e)

    def is_click_through(self) -> bool:
        return self._click_through

    # -- drag ---------------------------------------------------------
    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        try:
            if event.button() == Qt.LeftButton:
                try:
                    self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                except Exception:
                    # older PySide: globalPos()
                    self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
        except Exception:
            self._drag_pos = None

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        try:
            if self._drag_pos is not None and event.buttons() & Qt.LeftButton:
                try:
                    self.move(event.globalPosition().toPoint() - self._drag_pos)
                except Exception:
                    self.move(event.globalPos() - self._drag_pos)
        except Exception:
            pass

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        self._drag_pos = None

    def closeEvent(self, event) -> None:  # type: ignore[override]
        """Graceful shutdown: stop hotkeys and quit the app loop (no os._exit).

        main.py listens to signal_quit to stop audio/network threads; we notify
        it first, and only fall back to app.quit() when nobody else handles it.
        """
        try:
            self.stop_hotkeys()
        except Exception:
            pass
        try:
            notified = False
            try:
                notified = bool(self.isSignalConnected(self.signal_quit))  # type: ignore[arg-type]
            except Exception:
                notified = False
            if notified:
                try:
                    self.signal_quit.emit()
                except Exception:
                    pass
            else:
                app = QApplication.instance()
                if app is not None:
                    try:
                        app.quit()
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            if event is not None:
                event.accept()
        except Exception:
            pass

    # -- internal slots ------------------------------------------------
    def _toggle_visibility(self) -> None:
        try:
            if self.isVisible():
                self.hide()
            else:
                self.show()
                self._schedule_exclusion()
        except Exception:
            pass

    def _toggle_clickthrough(self) -> None:
        try:
            self.set_click_through(not self._click_through)
        except Exception:
            pass

    def _on_clear_btn(self) -> None:
        # single path: signal_clear -> clear_text (no double-clear)
        try:
            self.signal_clear.emit()
        except Exception:
            pass

    def _restore_copy_label(self) -> None:
        try:
            self.copy_btn.setText("Copy")
        except Exception:
            pass

    def _on_copy(self) -> None:
        try:
            cb = QApplication.clipboard()
            if cb is not None:
                cb.setText(self.viewer.toPlainText())
                self.copy_btn.setText("Copied!")
                try:
                    from PySide6.QtCore import QTimer

                    QTimer.singleShot(1200, self._restore_copy_label)
                except Exception:
                    pass
        except Exception:
            pass

    def _on_quit_btn(self) -> None:
        try:
            self.signal_quit.emit()
        except Exception:
            pass

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
            # graceful: owner (main.py) stops audio/network then quits.
            try:
                self.signal_quit.emit()
            except Exception:
                pass

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
                try:
                    self._hotkeys.stop()
                except Exception:
                    pass
        finally:
            self._hotkeys = None
