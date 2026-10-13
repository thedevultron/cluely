"""PyInstaller build script for InterviewCopilot (Windows .exe)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

APP_NAME = "InterviewCopilot"
ENTRY = "main.py"
ICON = "app_icon.ico"

HIDDEN_IMPORTS = [
    # PySide6 (only Qt binding — do NOT add PyQt6)
    "PySide6",
    "PySide6.QtCore",
    "PySide6.QtWidgets",
    "PySide6.QtGui",
    "PySide6.QtNetwork",
    "shiboken6",
    # audio / numeric / cloud
    "sounddevice",
    "soundcard",
    "numpy",
    "google.genai",
    "google.genai.types",
    "httpx",
    "pypdf",
    # hotkeys (win32 backends)
    "pynput",
    "pynput.keyboard",
    "pynput.keyboard._win32",
    "pynput.mouse",
    "pynput.mouse._win32",
    # local InterviewCopilot modules (ensure frozen imports resolve)
    "knowledge",
    "websearch",
    "audio_engine",
    "gemini_live_client",
    "config_manager",
    "overlay_window",
    "settings_dialog",
]

VERSION_INFO_TXT = """VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=(1, 0, 0, 0),
    prodvers=(1, 0, 0, 0),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo(
      [
        StringTable(
          '040904B0',
          [
            StringStruct('CompanyName', 'InterviewCopilot'),
            StringStruct('FileDescription', 'InterviewCopilot - real-time interview assistant'),
            StringStruct('FileVersion', '1.0.0'),
            StringStruct('InternalName', 'InterviewCopilot'),
            StringStruct('LegalCopyright', 'Copyright (c) InterviewCopilot'),
            StringStruct('OriginalFilename', 'InterviewCopilot.exe'),
            StringStruct('ProductName', 'InterviewCopilot'),
            StringStruct('ProductVersion', '1.0.0'),
          ]
        ),
      ]
    ),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
"""


def ensure_version_info(base: str) -> str | None:
    """Write truthful version_info.txt (never impersonates a system binary)."""
    path = os.path.join(base, "version_info.txt")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(VERSION_INFO_TXT)
        return path
    except OSError as e:
        print(f"warning: could not write version_info.txt: {e}", file=sys.stderr)
        return None


def build(onefile: bool = False, clean: bool = True) -> int:
    base = os.path.dirname(os.path.abspath(__file__))
    entry = os.path.join(base, ENTRY)
    if not os.path.exists(entry):
        print(f"Entry not found: {entry}", file=sys.stderr)
        return 2

    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller not installed. Run: pip install pyinstaller", file=sys.stderr)
        return 2

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--name", APP_NAME,
        "--windowed",  # no console window
    ]
    cmd.append("--onefile" if onefile else "--onedir")
    for h in HIDDEN_IMPORTS:
        cmd += ["--hidden-import", h]
    # PySide6 is our only Qt binding — exclude the OTHERS to avoid 500MB bundle.
    # Never exclude PySide6/shiboken6 here (that breaks the exe).
    for excluded in ("PySide2", "shiboken2", "PyQt5", "PyQt6", "PyQt6.QtWebEngine"):
        cmd += ["--exclude-module", excluded]
    # collect native binaries / data for audio + google SDK
    cmd += [
        "--collect-all", "sounddevice",
        "--collect-all", "soundcard",
        "--collect-all", "google",
    ]
    # optional icon (only if the file exists)
    icon_path = os.path.join(base, ICON)
    if os.path.exists(icon_path):
        cmd += ["--icon", icon_path]
    # truthful version resource on Windows
    if os.name == "nt":
        ver = ensure_version_info(base)
        if ver and os.path.exists(ver):
            cmd += ["--version-file", ver]
    if clean:
        cmd.append("--clean")
    cmd.append(entry)

    print("Running:", " ".join(cmd))
    rc = subprocess.call(cmd, cwd=base)
    if rc != 0:
        return rc

    dist = os.path.join(base, "dist", APP_NAME + (".exe" if os.name == "nt" else ""))
    target = os.path.join(base, "dist", APP_NAME + ".exe") if os.name == "nt" else dist
    print(f"\nBuild done. Output: {target}")
    print("Tip: test --onedir first for fast startup, then rebuild with --onefile.")
    print("Ship config.json alongside the exe (first run creates it if missing).")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Build InterviewCopilot Windows exe")
    p.add_argument("--onefile", action="store_true", help="single-file exe (slower boot)")
    p.add_argument("--no-clean", action="store_true", help="skip --clean")
    args = p.parse_args()
    return build(onefile=args.onefile, clean=not args.no_clean)


if __name__ == "__main__":
    raise SystemExit(main())
