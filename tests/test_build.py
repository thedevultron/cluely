"""Build-system tests: build_exe imports, hidden imports, version info, requirements.

Headless: never launches the GUI, never runs PyInstaller for real
(subprocess is monkeypatched to capture the command line instead).
"""
from __future__ import annotations

import os
import re
import sys

import pytest

# Make the project root importable regardless of the pytest invocation cwd.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import build_exe  # noqa: E402

# Local first-party modules that must appear in HIDDEN_IMPORTS so the frozen
# exe resolves its lazy imports (main.py delays overlay/settings/knowledge).
LOCAL_MODULES = [
    "audio_engine",
    "gemini_live_client",
    "config_manager",
    "knowledge",
    "overlay_window",
    "settings_dialog",
    "websearch",
]

# (package, minimum version) pairs that requirements.txt must pin.
REQUIRED_DEPS = {
    "PySide6": "6.6",
    "numpy": "1.26",
    "sounddevice": "0.4.6",
    "soundcard": "0.4.2",
    "google-genai": "1.0.0",
    "pynput": "1.7.6",
    "httpx": "0.27",
    "pypdf": "4.0",
    "pyinstaller": "6.0",
}


def test_build_exe_imports_and_constants():
    assert build_exe.APP_NAME == "InterviewCopilot"
    assert build_exe.ENTRY == "main.py"
    assert build_exe.ICON == "app_icon.ico"
    assert callable(build_exe.build)
    assert callable(build_exe.main)
    assert callable(build_exe.ensure_version_info)
    # Entry point and icon exist on disk.
    assert os.path.exists(os.path.join(ROOT, build_exe.ENTRY)), "main.py missing"
    assert os.path.exists(os.path.join(ROOT, build_exe.ICON)), "app_icon.ico missing"


def test_hidden_imports_contains_local_modules():
    for mod in LOCAL_MODULES:
        assert mod in build_exe.HIDDEN_IMPORTS, f"local module {mod!r} missing from HIDDEN_IMPORTS"


def test_hidden_imports_every_local_py_file_covered():
    """Every top-level .py module (except the build script + entry script,
    which PyInstaller uses as the Analysis target) is listed."""
    entry = build_exe.ENTRY  # main.py — the entry point, not a hidden import
    for fname in os.listdir(ROOT):
        if fname.endswith(".py") and fname not in ("build_exe.py", entry):
            mod = fname[: -len(".py")]
            assert mod in build_exe.HIDDEN_IMPORTS, f"{mod!r} (file {fname}) not in HIDDEN_IMPORTS"


def test_hidden_imports_pyside6_only():
    hiddens = build_exe.HIDDEN_IMPORTS
    assert "PySide6" in hiddens
    assert "PySide6.QtCore" in hiddens
    assert "PySide6.QtWidgets" in hiddens
    assert any(x.startswith("PyQt") for x in hiddens) is False, "PyQt must never be bundled"
    assert "PySide2" not in hiddens
    for required in ("sounddevice", "soundcard", "google.genai", "httpx", "pypdf", "pynput"):
        assert required in hiddens, f"{required!r} missing from HIDDEN_IMPORTS"


def test_version_info_contains_key_fields():
    txt = build_exe.VERSION_INFO_TXT
    for field in (
        "CompanyName",
        "FileDescription",
        "FileVersion",
        "InternalName",
        "LegalCopyright",
        "OriginalFilename",
        "ProductName",
        "ProductVersion",
        "Translation",
        "VSVersionInfo",
        "FixedFileInfo",
    ):
        assert field in txt, f"VERSION_INFO_TXT missing {field!r}"
    assert "InterviewCopilot" in txt
    # Must describe THIS app, never impersonate a system binary.
    for forbidden in ("Microsoft Corporation", "svchost", "Windows Operating System"):
        assert forbidden not in txt, f"VERSION_INFO_TXT impersonates a system binary ({forbidden})"


def test_version_info_file_matches_embedded(tmp_path):
    path = build_exe.ensure_version_info(str(tmp_path))
    assert path is not None
    written = open(path, encoding="utf-8").read()
    assert written == build_exe.VERSION_INFO_TXT


def test_requirements_pins():
    req_path = os.path.join(ROOT, "requirements.txt")
    text = open(req_path, encoding="utf-8").read()
    lines = [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    assert lines, "requirements.txt has no runtime deps"
    for ln in lines:
        assert not re.search(r"(?i)\bpyqt", ln), f"requirements.txt must not pull PyQt: {ln}"
    for pkg, min_ver in REQUIRED_DEPS.items():
        pat = re.compile(rf"(?i)^{re.escape(pkg)}\s*>=\s*(\d+(?:\.\d+)*)")
        found = [pat.match(ln) for ln in lines if pat.match(ln)]
        assert found, f"requirements.txt missing {pkg}>={min_ver}"
        assert found[0].group(1) == min_ver, f"{pkg} pinned to {found[0].group(1)}, expected >={min_ver}"


def _capture_build(monkeypatch, tmp_path, **kwargs):
    """Run build_exe.build with PyInstaller mocked; return (rc, cmd)."""
    captured = {}

    def fake_call(cmd, cwd=None):
        captured["cmd"] = list(cmd)
        captured["cwd"] = cwd
        return 0

    monkeypatch.setattr(build_exe.subprocess, "call", fake_call)
    monkeypatch.setattr(build_exe, "ensure_version_info", lambda base: None)
    rc = build_exe.build(**kwargs)
    return rc, captured.get("cmd", [])


def test_build_defaults_onedir_windowed(monkeypatch):
    rc, cmd = _capture_build(monkeypatch, None)
    assert rc == 0
    assert "--onedir" in cmd
    assert "--onefile" not in cmd
    assert "--windowed" in cmd
    assert "--clean" in cmd
    assert "--name" in cmd and "InterviewCopilot" in cmd
    assert cmd[0] == sys.executable and cmd[1:3] == ["-m", "PyInstaller"]
    assert cmd[-1].endswith("main.py")


def test_build_onefile_variant(monkeypatch):
    rc, cmd = _capture_build(monkeypatch, None, onefile=True)
    assert rc == 0
    assert "--onefile" in cmd
    assert "--onedir" not in cmd


def test_build_console_variant(monkeypatch):
    rc, cmd = _capture_build(monkeypatch, None, console=True)
    assert rc == 0
    assert "--windowed" not in cmd


def test_build_no_clean_variant(monkeypatch):
    rc, cmd = _capture_build(monkeypatch, None, clean=False)
    assert rc == 0
    assert "--clean" not in cmd


def test_build_hidden_imports_and_excludes(monkeypatch):
    rc, cmd = _capture_build(monkeypatch, None)
    assert rc == 0
    for mod in build_exe.HIDDEN_IMPORTS:
        assert "--hidden-import" in cmd
        assert mod in cmd
    for excluded in ("PySide2", "shiboken2", "PyQt5", "PyQt6", "PyQt6.QtWebEngine"):
        assert "--exclude-module" in cmd
        assert excluded in cmd
    # Native/data collection for audio + google SDK.
    for pkg in ("sounddevice", "soundcard", "google"):
        assert "--collect-all" in cmd
        assert pkg in cmd


def test_main_arg_parsing(monkeypatch):
    seen = {}

    def fake_build(onefile=False, console=False, clean=True):
        seen.update(onefile=onefile, console=console, clean=clean)
        return 0

    monkeypatch.setattr(build_exe, "build", fake_build)
    monkeypatch.setattr(sys, "argv", ["build_exe.py", "--onefile"])
    assert build_exe.main() == 0
    assert seen == {"onefile": True, "console": False, "clean": True}

    monkeypatch.setattr(sys, "argv", ["build_exe.py", "--console", "--no-clean"])
    assert build_exe.main() == 0
    assert seen == {"onefile": False, "console": True, "clean": False}
