"""PySide6 pre-launch settings dialog for InterviewCopilot."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtGui import QFont

from config_manager import DEFAULT_CONFIG

log = logging.getLogger("cluely.settings")

MODEL_CHOICES = [
    "gemini-3.8-live",
    "gemini-3.8-live-extended-thinking",
    "gemini-3.1-flash-live-preview",
    "gemini-2.5-flash-native-audio-preview-12-2025",
]

# Shown for extended-thinking (server-side reasoning) AND mapped to prompt
# for every model via difficulty (so difficulty always has an effect).
THINKING_CHOICES = ["MINIMAL", "LOW", "MEDIUM", "HIGH"]
THINKING_TIPS = {
    "MINIMAL": "Fastest, lowest latency. Factual hints only.",
    "LOW": "Quick bullets, light reasoning.",
    "MEDIUM": "Balanced reasoning / latency.",
    "HIGH": "Deepest reasoning. Best for system design / hard algo.",
}
DIFFICULTY_CHOICES = ["EASY", "MEDIUM", "HARD", "EXPERT"]
DIFFICULTY_TIPS = {
    "EASY": "Short simple hints, beginner-friendly. Low detail.",
    "MEDIUM": "Balanced concise bullets. Default.",
    "HARD": "Dense expert hints, edge cases + complexity.",
    "EXPERT": "Maximum depth: system-design grade, trade-offs + pitfalls.",
}

DARK_QSS = """
QDialog, QWidget { background: #1e1e1e; color: #e8e8e8; font-size: 13px; }
QTabWidget::pane { border: 1px solid #333; border-radius: 8px; background: #232323; }
QTabBar::tab { background: #2a2a2a; color: #bbb; padding: 8px 14px; border-top-left-radius: 8px; border-top-right-radius: 8px; margin-right: 2px; }
QTabBar::tab:selected { background: #323232; color: #2ecc71; }
QLineEdit, QPlainTextEdit, QComboBox, QSpinBox { background: #2b2b2b; border: 1px solid #3a3a3a; border-radius: 6px; padding: 6px; color: #eee; }
QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus { border: 1px solid #2ecc71; }
QListWidget { background: #2b2b2b; border: 1px solid #3a3a3a; border-radius: 6px; color: #eee; }
QPushButton { background: #2d2d2d; color: #eee; border: 1px solid #3d3d3d; border-radius: 8px; padding: 7px 14px; }
QPushButton:hover { background: #383838; border-color: #2ecc71; }
QPushButton:default { background: #2ecc71; color: #111; border: none; font-weight: bold; }
QPushButton:default:hover { background: #34e07f; }
QCheckBox { color: #ddd; spacing: 6px; }
QLabel { color: #ccc; }
QSlider::groove:horizontal { height: 6px; background: #333; border-radius: 3px; }
QSlider::handle:horizontal { width: 14px; margin: -5px 0; border-radius: 7px; background: #2ecc71; }
"""


DEVICE_QUERY_TIMEOUT_S = 8.0


def _call_timeout(fn, timeout_s: float = DEVICE_QUERY_TIMEOUT_S, default=None):
    """Run fn() with a hard timeout. Device APIs (WASAPI/PortAudio) can hang
    indefinitely on some machines — never let them freeze the UI. Never raises."""
    try:
        import concurrent.futures as _cf
        ex = _cf.ThreadPoolExecutor(max_workers=1)
        try:
            fut = ex.submit(fn)
            try:
                return fut.result(timeout=timeout_s)
            except _cf.TimeoutError:
                log.warning("device query timed out after %ss", timeout_s)
                try:
                    print(f"[devices] query timed out after {timeout_s}s, continuing without it",
                          flush=True)
                except Exception:
                    pass
                return default
        finally:
            try:
                ex.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
    except Exception as e:
        log.debug("device query failed: %s", e)
        return default


def _query_mics() -> List[Tuple[Optional[int], str]]:
    def _do() -> List[Tuple[Optional[int], str]]:
        try:
            from audio_engine import list_microphones
            mics = list_microphones()
            if mics:
                return [(idx, name) for idx, name in mics]
        except Exception as e:
            log.debug("mic query failed: %s", e)
        try:
            import sounddevice as sd
            out: List[Tuple[Optional[int], str]] = []
            for i, d in enumerate(sd.query_devices()):
                try:
                    if int(d.get("max_input_channels", 0)) > 0:
                        out.append((i, str(d.get("name", f"Device {i}"))))
                except Exception:
                    continue
            return out
        except Exception as e:
            log.debug("sounddevice fallback failed: %s", e)
            return []
    r = _call_timeout(_do, DEVICE_QUERY_TIMEOUT_S, [])
    return r if isinstance(r, list) else []


def _query_loopbacks() -> List[str]:
    # Canonical enumeration: WASAPI loopback mics only. Do NOT fall back to
    # all_speakers() names — those don't match get_microphone(loopback) lookup.
    def _do() -> List[str]:
        try:
            from audio_engine import list_loopbacks
            names = list_loopbacks()
            if names:
                return names
        except Exception as e:
            log.debug("loopback query failed: %s", e)
        return []
    r = _call_timeout(_do, DEVICE_QUERY_TIMEOUT_S, [])
    return r if isinstance(r, list) else []


class SettingsDialog(QDialog):
    launch_requested = Signal(dict)
    devices_ready = Signal(list, list)  # (mics, loopbacks) from background scan thread

    def __init__(self, config: Dict[str, Any], parent=None) -> None:
        super().__init__(parent, Qt.Dialog | Qt.Tool)
        self.setWindowTitle("Setup — System Audio Helper")
        # Stealth: Tool => hide settings from taskbar + Alt+Tab (WS_EX_TOOLWINDOW).
        self.setMinimumSize(640, 620)
        self.setStyleSheet(DARK_QSS)
        self._config = dict(config)
        # device selections requested before the background scan finishes
        self._pending_mic = config.get("selected_mic_index", None)
        self._pending_lb = str(config.get("selected_loopback_name", "") or "")
        self._devices_done = False
        try:
            self.devices_ready.connect(self._on_devices_ready)
        except Exception:
            pass

        layout = QVBoxLayout(self)
        self.stealth_banner = QLabel("", self)
        self.stealth_banner.setWordWrap(True)
        self.stealth_banner.setStyleSheet("color: #888; font-size: 11px; padding: 4px 2px;")
        layout.addWidget(self.stealth_banner)
        self.tabs = QTabWidget(self)
        layout.addWidget(self.tabs)

        self._build_api_tab()
        self._build_audio_tab()
        self._build_knowledge_tab()
        self._build_prompt_tab()
        self._build_controls_tab()

        btn_row = QHBoxLayout()
        self.btn_save = QPushButton("Save")
        self.btn_launch = QPushButton("▶  Save && Launch")
        self.btn_launch.setDefault(True)
        self.btn_cancel = QPushButton("Cancel")
        btn_row.addWidget(self.btn_save)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_launch)
        layout.addLayout(btn_row)

        self.btn_save.clicked.connect(self._on_save)
        self.btn_launch.clicked.connect(self._on_launch)
        self.btn_cancel.clicked.connect(self.reject)

        self._load_from_config(self._config)
        # stealth: hide setup from capture too (API key / docs are sensitive)
        try:
            from PySide6.QtCore import QTimer as _QTimer
            _QTimer.singleShot(150, self._apply_stealth)
        except Exception:
            pass
        # devices: scan in background so a hanging WASAPI/PortAudio API can
        # never freeze the dialog. Combos fill in when the scan lands.
        try:
            self.audio_status_label.setText("Scanning audio devices…")
            import threading as _th
            _th.Thread(target=self._scan_devices_worker, name="dev-scan", daemon=True).start()
        except Exception:
            pass

    def _scan_devices_worker(self) -> None:
        try:
            mics = _query_mics()
        except Exception:
            mics = []
        try:
            lbs = _query_loopbacks()
        except Exception:
            lbs = []
        try:
            self.devices_ready.emit(list(mics or []), list(lbs or []))
        except Exception:
            pass

    def _on_devices_ready(self, mics, lbs) -> None:
        # Runs on the GUI thread (signal). Populate both combos, restore selection.
        try:
            self._devices_done = True
            cur_mic = self._pending_mic if self._pending_mic is not None else self.mic_combo.currentData()
            self.mic_combo.blockSignals(True)
            self.mic_combo.clear()
            self.mic_combo.addItem("Default input", None)
            for idx, name in (mics or []):
                try:
                    self.mic_combo.addItem(f"[{idx}] {name}", idx)
                except Exception:
                    continue
            if cur_mic is not None:
                try:
                    i = self.mic_combo.findData(int(cur_mic))
                except (TypeError, ValueError):
                    i = -1
                self.mic_combo.setCurrentIndex(i if i >= 0 else 0)
            else:
                self.mic_combo.setCurrentIndex(0)
            self.mic_combo.blockSignals(False)
        except Exception:
            try:
                self.mic_combo.blockSignals(False)
            except Exception:
                pass
        try:
            preserve = self._pending_lb
            if not preserve:
                try:
                    cur_data = self.loopback_combo.currentData()
                    cur_text = self.loopback_combo.currentText()
                    if cur_data:
                        preserve = str(cur_data)
                    elif cur_text and "Auto-detect" not in cur_text and "Scanning" not in cur_text:
                        preserve = cur_text
                except Exception:
                    preserve = ""
            self._reload_loopbacks(list(lbs or []), preserve=preserve or "")
            self._pending_mic = None
            self._pending_lb = ""
        except Exception:
            pass

    def showEvent(self, event) -> None:  # type: ignore[override]
        try:
            super().showEvent(event)
        except Exception:
            pass
        try:
            # Unmissable: Tool windows have no taskbar/Alt+Tab entry, so center
            # on the primary screen and bring to front on first show.
            if not getattr(self, "_centered", False):
                self._centered = True
                try:
                    from PySide6.QtWidgets import QApplication as _QA
                    scr = _QA.primaryScreen()
                    if scr is not None:
                        geo = scr.availableGeometry()
                        self.move(geo.center() - self.rect().center())
                except Exception:
                    pass
                try:
                    self.raise_()
                    self.activateWindow()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            from PySide6.QtCore import QTimer as _QTimer
            _QTimer.singleShot(150, self._apply_stealth)
        except Exception:
            pass

    def _apply_stealth(self) -> None:
        try:
            from overlay_window import apply_capture_exclusion, get_capture_exclusion_status
            ok = apply_capture_exclusion(self)
            try:
                verified = get_capture_exclusion_status(self)
            except Exception:
                verified = ok
            if verified:
                self.stealth_banner.setText("🛡️ Stealth ON — setup is hidden from screenshots & screenshare, and not in Alt+Tab.")
                self.stealth_banner.setStyleSheet("color: #2ecc71; font-size: 11px; padding: 4px 2px;")
            else:
                self.stealth_banner.setText("⚠️ Stealth OFF — screenshots may include this window (needs Win10 2004+). HUD protection is separate.")
                self.stealth_banner.setStyleSheet("color: #ff6b6b; font-size: 11px; padding: 4px 2px;")
        except Exception:
            pass

    # -- tabs ---------------------------------------------------------
    def _build_api_tab(self) -> None:
        w = QWidget(self)
        form = QFormLayout(w)

        self.api_key_edit = QLineEdit(w)
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_key_edit.setPlaceholderText("AIza... (Google AI Studio key)")
        reveal = QCheckBox("Show", w)
        reveal.toggled.connect(
            lambda on: self.api_key_edit.setEchoMode(
                QLineEdit.Normal if on else QLineEdit.Password
            )
        )
        key_row = QHBoxLayout()
        key_row.addWidget(self.api_key_edit, 1)
        key_row.addWidget(reveal)
        key_wrap = QWidget(w)
        key_wrap.setLayout(key_row)
        form.addRow("API Key:", key_wrap)

        self.model_combo = QComboBox(w)
        self.model_combo.setEditable(True)
        self.model_combo.addItems(MODEL_CHOICES)
        form.addRow("Live model:", self.model_combo)

        self.fallback_combo = QComboBox(w)
        self.fallback_combo.setEditable(True)
        self.fallback_combo.addItems(MODEL_CHOICES)
        form.addRow("Fallback Live model:", self.fallback_combo)

        self.summarizer_combo = QComboBox(w)
        self.summarizer_combo.setEditable(True)
        self.summarizer_combo.addItems(MODEL_CHOICES)
        form.addRow("Summarizer model:", self.summarizer_combo)

        # Thinking level — works on extended-thinking model (server-side);
        # difficulty below always works via prompt mapping on every model.
        self.thinking_wrap = QWidget(w)
        thinking_layout = QHBoxLayout(self.thinking_wrap)
        thinking_layout.setContentsMargins(0, 0, 0, 0)
        self.thinking_combo = QComboBox(self.thinking_wrap)
        self.thinking_combo.addItems(THINKING_CHOICES)
        self.thinking_combo.setToolTip("Server reasoning depth (extended-thinking model).")
        self.thinking_tip = QLabel("", self.thinking_wrap)
        self.thinking_tip.setWordWrap(True)
        self.thinking_tip.setStyleSheet("color: #888; font-size: 11px;")
        thinking_layout.addWidget(self.thinking_combo)
        thinking_layout.addWidget(self.thinking_tip, 1)
        self.thinking_label = QLabel("Reasoning depth:", w)
        self.thinking_label.setToolTip("Only the *-extended-thinking model uses this server-side.")
        form.addRow(self.thinking_label, self.thinking_wrap)
        self.thinking_combo.currentTextChanged.connect(self._on_thinking_changed)

        self.difficulty_combo = QComboBox(w)
        self.difficulty_combo.addItems(DIFFICULTY_CHOICES)
        self.difficulty_combo.setToolTip("Works on EVERY model — controls answer depth via prompt.")
        self.difficulty_tip = QLabel("", w)
        self.difficulty_tip.setWordWrap(True)
        self.difficulty_tip.setStyleSheet("color: #2ecc71; font-size: 11px;")
        diff_wrap = QWidget(w)
        diff_layout = QHBoxLayout(diff_wrap)
        diff_layout.setContentsMargins(0, 0, 0, 0)
        diff_layout.addWidget(self.difficulty_combo)
        diff_layout.addWidget(self.difficulty_tip, 1)
        form.addRow("Difficulty:", diff_wrap)
        self.difficulty_combo.currentTextChanged.connect(
            lambda t: self.difficulty_tip.setText(DIFFICULTY_TIPS.get(str(t).upper(), ""))
        )
        self.model_combo.currentTextChanged.connect(self._on_model_changed)

        self.role_edit = QLineEdit(w)
        self.role_edit.setPlaceholderText("e.g. Backend Engineer")
        form.addRow("Target role:", self.role_edit)

        self.company_edit = QLineEdit(w)
        self.company_edit.setPlaceholderText("e.g. Acme Corp (optional)")
        form.addRow("Company:", self.company_edit)

        self.save_transcript_check = QCheckBox("Save transcript locally (.md)", w)
        form.addRow("", self.save_transcript_check)

        self.tabs.addTab(w, "API & Model")

    def _build_audio_tab(self) -> None:
        from PySide6.QtWidgets import QRadioButton, QButtonGroup
        w = QWidget(self)
        form = QFormLayout(w)

        info = QLabel(
            "Default: <b>computer sound only</b> — auto-detects your output device. "
            "No input device to pick. Just press Launch.",
            w,
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #2ecc71; font-size: 12px;")
        form.addRow("", info)

        mode_row = QHBoxLayout()
        self.radio_system = QRadioButton("Computer sound only (recommended)", w)
        self.radio_dual = QRadioButton("Mic + computer sound", w)
        self.radio_system.setChecked(True)
        self.radio_system.setToolTip("Capture only system/loopback audio (WASAPI). Mic is off.")
        self.radio_dual.setToolTip("Also capture your microphone (needs mic permission).")
        mode_row.addWidget(self.radio_system)
        mode_row.addWidget(self.radio_dual)
        mode_wrap = QWidget(w)
        mode_wrap.setLayout(mode_row)
        form.addRow("Source:", mode_wrap)
        self._audio_group = QButtonGroup(w)
        self._audio_group.addButton(self.radio_system)
        self._audio_group.addButton(self.radio_dual)
        try:
            self.radio_system.toggled.connect(self._on_audio_mode_changed)
        except Exception:
            pass

        self.mic_combo = QComboBox(w)
        self.mic_combo.addItem("Default input", None)
        self.mic_combo.addItem("Scanning devices…", "__scan__")
        self.mic_combo.setEnabled(False)
        form.addRow("Microphone:", self.mic_combo)

        self.loopback_combo = QComboBox(w)
        self.loopback_combo.setEditable(True)
        self.audio_status_label = QLabel("Scanning audio devices…", w)
        self.audio_status_label.setWordWrap(True)
        self.audio_status_label.setStyleSheet("color: #888; font-size: 11px;")
        self.loopback_combo.addItem("✨ Auto-detect (recommended)", "")
        form.addRow("System sound:", self.loopback_combo)
        form.addRow("", self.audio_status_label)

        vad_row = QHBoxLayout()
        self.vad_slider = QSlider(w)
        self.vad_slider.setOrientation(Qt.Horizontal)
        self.vad_slider.setRange(50, 2000)
        self.vad_slider.setSingleStep(10)
        self.vad_label = QLabel("300", w)
        self.vad_label.setMinimumWidth(50)
        self.vad_slider.valueChanged.connect(lambda v: self.vad_label.setText(str(v)))
        vad_row.addWidget(self.vad_slider, 1)
        vad_row.addWidget(self.vad_label)
        vad_wrap = QWidget(w)
        vad_wrap.setLayout(vad_row)
        form.addRow("Sensitivity:", vad_wrap)
        form.addRow("", QLabel("Higher = stricter silence cut. Lower = more sensitive.", w))

        refresh = QPushButton("↻ Refresh devices", w)
        refresh.clicked.connect(self._refresh_devices)
        form.addRow("", refresh)

        self.tabs.addTab(w, "Audio (auto)")

    def _reload_loopbacks(self, lbs: list | None = None, preserve: str = "") -> None:
        try:
            if lbs is None:
                lbs = _query_loopbacks()
        except Exception:
            lbs = []
        try:
            self.loopback_combo.blockSignals(True)
            self.loopback_combo.clear()
            # Item 0 = auto-detect (stores "")
            self.loopback_combo.addItem("✨ Auto-detect (recommended)", "")
            for n in lbs:
                self.loopback_combo.addItem(n, n)
            # restore selection
            target = (preserve or "").strip()
            if target:
                idx = self.loopback_combo.findData(target)
                if idx < 0:
                    idx = self.loopback_combo.findText(target)
                self.loopback_combo.setCurrentIndex(idx if idx >= 0 else 0)
            else:
                self.loopback_combo.setCurrentIndex(0)
            if lbs:
                auto = lbs[0]
                self.audio_status_label.setText(f"✅ Found {len(lbs)} device(s). Auto will use: {auto}")
                self.audio_status_label.setStyleSheet("color: #2ecc71; font-size: 11px;")
            else:
                self.audio_status_label.setText("⚠️ No system-audio device found. Check output device / soundcard install.")
                self.audio_status_label.setStyleSheet("color: #ff6b6b; font-size: 11px;")
        finally:
            try:
                self.loopback_combo.blockSignals(False)
            except Exception:
                pass

    def _on_audio_mode_changed(self) -> None:
        try:
            dual = self.radio_dual.isChecked()
            self.mic_combo.setEnabled(dual)
        except Exception:
            pass

    def _build_knowledge_tab(self) -> None:
        w = QWidget(self)
        v = QVBoxLayout(w)

        self.enable_knowledge_check = QCheckBox("Enable local knowledge base", w)
        self.enable_knowledge_check.setToolTip("Inject your files (resume, notes) into the system prompt.")
        v.addWidget(self.enable_knowledge_check)

        v.addWidget(QLabel("Knowledge files / folders:", w))
        self.knowledge_list = QListWidget(w)
        self.knowledge_list.setMinimumHeight(140)
        v.addWidget(self.knowledge_list, 1)

        btn_row = QHBoxLayout()
        self.btn_add_files = QPushButton("Add files", w)
        self.btn_add_folder = QPushButton("Add folder", w)
        self.btn_remove_path = QPushButton("Remove", w)
        self.btn_clear_paths = QPushButton("Clear", w)
        btn_row.addWidget(self.btn_add_files)
        btn_row.addWidget(self.btn_add_folder)
        btn_row.addWidget(self.btn_remove_path)
        btn_row.addWidget(self.btn_clear_paths)
        v.addLayout(btn_row)
        self.btn_add_files.clicked.connect(self._on_add_files)
        self.btn_add_folder.clicked.connect(self._on_add_folder)
        self.btn_remove_path.clicked.connect(self._on_remove_path)
        self.btn_clear_paths.clicked.connect(lambda: self.knowledge_list.clear())

        max_row = QHBoxLayout()
        max_row.addWidget(QLabel("Max chars:", w))
        self.knowledge_max_spin = QSpinBox(w)
        self.knowledge_max_spin.setRange(1000, 60000)
        self.knowledge_max_spin.setSingleStep(1000)
        max_row.addWidget(self.knowledge_max_spin)
        max_row.addStretch(1)
        v.addLayout(max_row)

        self.enable_web_check = QCheckBox("Enable web search (function calling)", w)
        v.addWidget(self.enable_web_check)
        v.addWidget(QLabel("Free unlimited DuckDuckGo, no key, no browser.", w))

        web_row = QHBoxLayout()
        web_row.addWidget(QLabel("Max results:", w))
        self.web_max_spin = QSpinBox(w)
        self.web_max_spin.setRange(1, 10)
        web_row.addWidget(self.web_max_spin)
        web_row.addStretch(1)
        v.addLayout(web_row)

        self.tabs.addTab(w, "Knowledge & Search")

    def _build_prompt_tab(self) -> None:
        w = QWidget(self)
        v = QVBoxLayout(w)
        v.addWidget(QLabel("System instruction sent to Gemini Live:", w))
        self.prompt_edit = QPlainTextEdit(w)
        self.prompt_edit.setMinimumHeight(220)
        v.addWidget(self.prompt_edit, 1)

        row = QHBoxLayout()
        self.font_spin = QSpinBox(w)
        self.font_spin.setRange(8, 28)
        row.addWidget(QLabel("Font:", w))
        row.addWidget(self.font_spin)
        row.addWidget(QLabel("Opacity:", w))
        self.opacity_slider = QSlider(w)
        self.opacity_slider.setOrientation(Qt.Horizontal)
        self.opacity_slider.setRange(30, 100)
        self.opacity_label = QLabel("85%", w)
        self.opacity_label.setMinimumWidth(42)
        self.opacity_slider.valueChanged.connect(
            lambda v: self.opacity_label.setText(f"{int(v)}%")
        )
        row.addWidget(self.opacity_slider, 1)
        row.addWidget(self.opacity_label)
        self.auto_resize_check = QCheckBox("Auto-resize HUD", w)
        self.auto_resize_check.setToolTip("Grow HUD height with content (up to 60% of screen).")
        row.addWidget(self.auto_resize_check)
        v.addLayout(row)

        row2 = QHBoxLayout()
        row2.addStretch(1)
        self.btn_reset_prompt = QPushButton("Reset to Default", w)
        self.btn_reset_prompt.clicked.connect(self._on_reset_prompt)
        row2.addWidget(self.btn_reset_prompt)
        v.addLayout(row2)
        self.tabs.addTab(w, "Prompt & HUD")

    def _build_controls_tab(self) -> None:
        w = QWidget(self)
        v = QVBoxLayout(w)
        legend = QLabel(
            "<b>Just press Launch.</b> The HUD appears on your screen — no input device to pick, "
            "system sound is auto-detected.<br><br>"
            "<b>Global shortcuts</b> (work while HUD is unfocused):<br><br>"
            "<b>Ctrl+Alt+H</b> — Hide / Show HUD<br>"
            "<b>Ctrl+Alt+C</b> — Clear HUD text<br>"
            "<b>Ctrl+Alt+P</b> — Pause / Resume stream (Context Saver)<br>"
            "<b>Ctrl+Alt+T</b> — Toggle click-through<br>"
            "<b>Ctrl+Alt+Q</b> — Kill process immediately<br><br>"
            "🛡️ HUD is pinned on top, hidden from screenshots & screenshare on Win10 2004+ "
            "(WDA_EXCLUDEFROMCAPTURE — see STEALTH pill on HUD), and never appears in Alt+Tab "
            "or the taskbar. Setup window is also capture-excluded.",
            w,
        )
        legend.setWordWrap(True)
        v.addWidget(legend)
        v.addStretch(1)
        self.tabs.addTab(w, "Controls")

    # -- logic --------------------------------------------------------
    def _on_model_changed(self, text: str) -> None:
        """Reasoning depth always visible; hint explains per-model effect."""
        try:
            is_thinking_model = "extended-thinking" in str(text).lower()
            # Always keep both rows visible — difficulty works on every model.
            try:
                self.thinking_wrap.setVisible(True)
                self.thinking_label.setVisible(True)
            except Exception:
                pass
            if hasattr(self, "thinking_tip"):
                if is_thinking_model:
                    self._on_thinking_changed(self.thinking_combo.currentText())
                else:
                    self.thinking_tip.setText(
                        "Plain models ignore server reasoning — use Difficulty below (always works)."
                    )
        except Exception:
            pass

    def _on_thinking_changed(self, text: str) -> None:
        try:
            self.thinking_tip.setText(THINKING_TIPS.get(str(text).upper(), ""))
        except Exception:
            pass

    def _on_add_files(self) -> None:
        try:
            files, _ = QFileDialog.getOpenFileNames(
                self, "Add knowledge files", os.path.expanduser("~"),
                "Text files (*.txt *.md *.py *.json *.csv *.pdf);;All files (*.*)",
            )
            for f in files or []:
                try:
                    ap = os.path.abspath(f)
                    if self.knowledge_list.findItems(ap, Qt.MatchExactly):
                        continue
                    self.knowledge_list.addItem(ap)
                except Exception:
                    continue
        except Exception as e:
            QMessageBox.warning(self, "Knowledge", f"Add files failed:\n{e}")

    def _on_add_folder(self) -> None:
        try:
            folder = QFileDialog.getExistingDirectory(self, "Add knowledge folder", os.path.expanduser("~"))
            if folder:
                try:
                    ap = os.path.abspath(folder)
                    if not self.knowledge_list.findItems(ap, Qt.MatchExactly):
                        self.knowledge_list.addItem(ap)
                except Exception:
                    pass
        except Exception as e:
            QMessageBox.warning(self, "Knowledge", f"Add folder failed:\n{e}")

    def _on_remove_path(self) -> None:
        try:
            for item in self.knowledge_list.selectedItems():
                try:
                    self.knowledge_list.takeItem(self.knowledge_list.row(item))
                except Exception:
                    continue
        except Exception:
            pass

    def _refresh_devices(self) -> None:
        # Re-scan in background (button never blocks on hanging audio APIs).
        try:
            self.audio_status_label.setText("Scanning audio devices…")
            self.audio_status_label.setStyleSheet("color: #888; font-size: 11px;")
            try:
                self._pending_mic = self.mic_combo.currentData()
            except Exception:
                pass
            try:
                cur_data = self.loopback_combo.currentData()
                cur_text = self.loopback_combo.currentText()
                if cur_data:
                    self._pending_lb = str(cur_data)
                elif cur_text and "Auto-detect" not in cur_text and "Scanning" not in cur_text:
                    self._pending_lb = cur_text
                else:
                    self._pending_lb = ""
            except Exception:
                self._pending_lb = ""
            import threading as _th
            _th.Thread(target=self._scan_devices_worker, name="dev-scan", daemon=True).start()
        except Exception as e:
            QMessageBox.warning(self, "Devices", f"Refresh failed:\n{e}")

    def _on_reset_prompt(self) -> None:
        self.prompt_edit.setPlainText(DEFAULT_CONFIG["system_instruction"])

    def _load_from_config(self, cfg: Dict[str, Any]) -> None:
        try:
            self.api_key_edit.setText(str(cfg.get("api_key", "")))
            self._set_combo_text(self.model_combo, str(cfg.get("model", "gemini-3.8-live")))
            self._set_combo_text(self.fallback_combo, str(cfg.get("fallback_model", "gemini-2.5-flash-native-audio-preview-12-2025")))
            self._set_combo_text(self.summarizer_combo, str(cfg.get("summarizer_model", "gemini-3.5-flash-lite")))
            self._set_combo_text(self.thinking_combo, str(cfg.get("thinking_level", "HIGH")).upper())
            self._on_thinking_changed(self.thinking_combo.currentText())
            try:
                self._set_combo_text(self.difficulty_combo, str(cfg.get("difficulty_level", "MEDIUM")).upper())
                self.difficulty_tip.setText(DIFFICULTY_TIPS.get(str(cfg.get("difficulty_level", "MEDIUM")).upper(), ""))
            except Exception:
                pass
            self.role_edit.setText(str(cfg.get("target_role", "")))
            self.company_edit.setText(str(cfg.get("target_company", "")))
            self.save_transcript_check.setChecked(bool(cfg.get("save_transcript_locally", True)))

            # audio mode (system-only default)
            try:
                mode = str(cfg.get("audio_mode", "system") or "system").lower()
                if mode == "dual":
                    self.radio_dual.setChecked(True)
                else:
                    self.radio_system.setChecked(True)
                self._on_audio_mode_changed()
            except Exception:
                pass
            mic_idx = cfg.get("selected_mic_index", None)
            # actual combo fill happens in _on_devices_ready (background scan);
            # remember the request so the selection is restored when devices land.
            self._pending_mic = mic_idx
            try:
                self._pending_lb = str(cfg.get("selected_loopback_name", "") or "")
            except Exception:
                self._pending_lb = ""

            try:
                self.vad_slider.setValue(int(cfg.get("vad_threshold_db", 300)))
            except (TypeError, ValueError):
                self.vad_slider.setValue(300)
            # knowledge & search
            try:
                self.enable_knowledge_check.setChecked(bool(cfg.get("enable_knowledge", False)))
            except Exception:
                pass
            try:
                self.knowledge_list.clear()
                for p in (cfg.get("knowledge_paths", []) or []):
                    try:
                        if str(p).strip():
                            self.knowledge_list.addItem(os.path.abspath(str(p)))
                    except Exception:
                        continue
            except Exception:
                pass
            try:
                self.knowledge_max_spin.setValue(int(cfg.get("knowledge_max_chars", 12000)))
            except (TypeError, ValueError):
                self.knowledge_max_spin.setValue(12000)
            try:
                self.enable_web_check.setChecked(bool(cfg.get("enable_web_search", False)))
            except Exception:
                pass
            try:
                self.web_max_spin.setValue(int(cfg.get("web_max_results", 5)))
            except (TypeError, ValueError):
                self.web_max_spin.setValue(5)
            # prompt & HUD
            self.prompt_edit.setPlainText(str(cfg.get("system_instruction", "")))
            try:
                self.font_spin.setValue(int(cfg.get("font_size", 14)))
            except (TypeError, ValueError):
                self.font_spin.setValue(14)
            try:
                op = float(cfg.get("hud_opacity", 0.85))
                self.opacity_slider.setValue(int(round(max(0.3, min(1.0, op)) * 100)))
            except (TypeError, ValueError):
                self.opacity_slider.setValue(85)
            try:
                self.auto_resize_check.setChecked(bool(cfg.get("hud_auto_resize", True)))
            except Exception:
                self.auto_resize_check.setChecked(True)
            # apply thinking-row visibility after model is set
            try:
                self._on_model_changed(self.model_combo.currentText())
            except Exception:
                pass
        except Exception as e:
            log.warning("load into dialog failed: %s", e)

    @staticmethod
    def _set_combo_text(combo: QComboBox, text: str) -> None:
        i = combo.findText(text)
        if i >= 0:
            combo.setCurrentIndex(i)
        else:
            combo.setCurrentText(text)

    def collect_config(self) -> Dict[str, Any]:
        cfg = dict(self._config)
        thinking = self.thinking_combo.currentText().strip().upper() if hasattr(self, "thinking_combo") else "HIGH"
        if thinking in ("EXTENDED", "MAX", "ULTRA"):
            thinking = "HIGH"
        if thinking not in ("MINIMAL", "LOW", "MEDIUM", "HIGH"):
            thinking = "HIGH"
        try:
            difficulty = self.difficulty_combo.currentText().strip().upper()
        except Exception:
            difficulty = "MEDIUM"
        if difficulty not in ("EASY", "MEDIUM", "HARD", "EXPERT"):
            difficulty = "MEDIUM"
        try:
            dual = bool(self.radio_dual.isChecked())
        except Exception:
            dual = False
        audio_mode = "dual" if dual else "system"
        # loopback combo: data "" = auto-detect
        try:
            lb_data = self.loopback_combo.currentData()
            lb_text = self.loopback_combo.currentText().strip()
            if lb_data:
                loopback_name = str(lb_data).strip()
            elif lb_text and "Auto-detect" in lb_text:
                loopback_name = ""
            else:
                loopback_name = lb_text
        except Exception:
            loopback_name = ""
        knowledge_paths: List[str] = []
        try:
            for i in range(self.knowledge_list.count()):
                try:
                    t = self.knowledge_list.item(i).text()
                    if t and t.strip():
                        knowledge_paths.append(os.path.abspath(t.strip()))
                except Exception:
                    continue
        except Exception:
            knowledge_paths = []
        try:
            _mic_data = self.mic_combo.currentData()
            if _mic_data == "__scan__" and not getattr(self, "_devices_done", False):
                _mic_data = getattr(self, "_pending_mic", None)
            try:
                mic_idx_out = None if _mic_data is None else int(_mic_data)
            except (TypeError, ValueError):
                mic_idx_out = None
        except Exception:
            mic_idx_out = None
        cfg.update({
            "api_key": self.api_key_edit.text().strip(),
            "model": self.model_combo.currentText().strip() or "gemini-3.8-live",
            "fallback_model": self.fallback_combo.currentText().strip() or "gemini-2.5-flash-native-audio-preview-12-2025",
            "summarizer_model": self.summarizer_combo.currentText().strip() or "gemini-3.5-flash-lite",
            "thinking_level": thinking or "HIGH",
            "difficulty_level": difficulty,
            "audio_mode": audio_mode,
            "mic_enabled": bool(dual),
            "target_role": self.role_edit.text().strip(),
            "target_company": self.company_edit.text().strip(),
            "save_transcript_locally": self.save_transcript_check.isChecked(),
            "selected_mic_index": mic_idx_out,
            "selected_loopback_name": loopback_name,
            "vad_threshold_db": int(self.vad_slider.value()),
            "font_size": int(self.font_spin.value()),
            "hud_opacity": float(int(self.opacity_slider.value())) / 100.0,
            "system_instruction": self.prompt_edit.toPlainText(),
            "enable_knowledge": bool(self.enable_knowledge_check.isChecked()),
            "knowledge_paths": knowledge_paths,
            "knowledge_max_chars": int(self.knowledge_max_spin.value()),
            "enable_web_search": bool(self.enable_web_check.isChecked()),
            "web_max_results": int(self.web_max_spin.value()),
            "hud_auto_resize": bool(self.auto_resize_check.isChecked()),
        })
        # preserve ultimate fallback if present
        cfg.setdefault("ultimate_fallback_model", "gemini-2.5-flash-native-audio-preview-12-2025")
        return cfg

    def _on_save(self) -> None:
        try:
            from config_manager import save_config
            save_config(self.collect_config())
            QMessageBox.information(self, "InterviewCopilot", "Settings saved to config.json.")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", f"Could not save config:\n{e}")

    def _on_launch(self) -> None:
        data = self.collect_config()
        if not data.get("api_key"):
            r = QMessageBox.warning(
                self, "Missing API key",
                "API key is empty. Launch anyway?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if r != QMessageBox.Yes:
                return
        try:
            from config_manager import save_config
            save_config(data)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", f"Could not save config:\n{e}")
            return
        self.launch_requested.emit(data)
        self.accept()
