# InterviewCopilot — Deployment Checklist

End-to-end runbook: install → configure → run → build → smoke-test → ship.
See `architecture.html` for the full architecture (threading, queues, config schema).

## 0. Prerequisites

- [ ] Windows 10 2004+ (build 19041+) — required for `WDA_EXCLUDEFROMCAPTURE` stealth.
- [ ] Python 3.14 (3.11+ fine) — check: `python --version`.
- [ ] A Google AI Studio API key (`AIza...`).
- [ ] An output device (speakers/headphones) — system audio is captured via WASAPI loopback.

> PySide6 **only**. Never `pip install PyQt6` into this env for the app — dual Qt bindings
> bloat the exe to ~500 MB. The build excludes them; `requirements.txt` doesn't list them.

## 1. Install

```powershell
cd C:\Users\Amir\cluely
python -m pip install -r requirements.txt
python -m pip install pytest        # dev/test only — not a runtime dep
```

Verify pins (must all say OK, no PyQt lines):

```powershell
python -c "import re; t=open('requirements.txt').read(); print([l for l in t.splitlines() if l.strip() and not l.strip().startswith('#')])"
python -m py_compile main.py build_exe.py
```

## 2. Configure

- [ ] Run once: `python main.py` → settings dialog → paste API key → **Save & Launch**.
  First run creates `config.json` (beside `main.py` in source mode, beside the exe when frozen).
- [ ] Or create `config.json` manually — see the 26-key schema in `architecture.html` §7.
- [ ] Check the **Audio (auto)** tab shows a loopback device. `Selected system sound: (auto)`
  is fine if your default output is listed.
- [ ] (Optional) Knowledge & Search tab: add folders/files, enable web search.

⚠️ `config.json` contains your live API key. It is `.gitignore`d — never commit it, never
paste it into bug reports. Logs only ever show a masked fragment (`AIza...90 (len=39)`).

## 3. Run (source)

```powershell
python main.py          # settings dialog → HUD
python main.py --auto   # skip settings, straight to HUD (saved config)
python main.py --diag   # headless check: masked config + device probe + 60 s Live hello
```

Expected `--diag` output: `[diag] RESULT: PASS` (exit code 0). Play some computer audio
(music/video/meeting) during the 6 s audio probe if you want non-zero frame counts.

## 4. Build

```powershell
python build_exe.py             # onedir (default, fast startup) → dist\InterviewCopilot\
python build_exe.py --onefile   # single exe → dist\InterviewCopilot.exe
python build_exe.py --console   # keep a console (needed for frozen-exe --diag output)
```

- [ ] `app_icon.ico` present (it is, ~17 KB). If missing, the build warns and continues.
- [ ] `version_info.txt` regenerated automatically (1.0.0, CompanyName=InterviewCopilot).
- [ ] `InterviewCopilot.spec` regenerated automatically — don't hand-edit it (absolute paths).

### Installer (optional, needs Inno Setup 6)

1. Build first (onedir).
2. Open `installer.iss` in Inno Setup 6 → **Compile (Ctrl+F9)**.
3. Setup exe lands in `dist\installer\InterviewCopilot-Setup-1.0.0.exe`.
4. Bump `MyAppVersion` in `installer.iss` together with `build_exe.VERSION_INFO_TXT` /
   `version_info.txt` on release.

## 5. Smoke tests

Source mode:

```powershell
python -m pytest tests/test_build.py tests/test_main.py -v   # 26 passed expected
python main.py --diag                                        # RESULT: PASS
```

Frozen exe:

- [ ] Launch (double-click or `.\dist\InterviewCopilot\InterviewCopilot.exe`) → HUD appears,
  welcome line shows `model (text-only)`.
- [ ] Within ~12 s the HUD shows `Hearing audio ✔ (N frames sent)` (play any sound).
- [ ] Model replies stream as text; `Ctrl+Alt+H` hides/shows; `Ctrl+Alt+Q` quits.
- [ ] Console build only: `InterviewCopilot.exe --diag` prints to the terminal.
- [ ] Quit via hotkey/pill → no orphan processes in Task Manager; `client.stop()` + `engine.stop()`.

Failure looks like: `RESULT: FAIL` (diag), no text in ~60 s, or the HUD error line
`[connection issue: ...]` throttled at 1/5 s. Paste the **masked** `[diag]`/`DIAG` block
from `app.log` — never the raw key.

## 6. Ship

- [ ] Copy `dist\InterviewCopilot\` (onedir: the whole folder) **plus** a **blank/template**
  `config.json` (key removed). Never ship your real `config.json`.
- [ ] (onefile) Copy `dist\InterviewCopilot.exe` + template `config.json` side by side.
- [ ] Keep `transcripts/` empty in the shipped tree (runtime-generated; not gitignored by
  default — clean before zipping).

## 7. Secret / hygiene check (already verified)

- [x] `.gitignore` covers `config.json`, `.env`, `.env.*`, `app.log`, `*.log`, `dist/`, `build/`.
- [x] `git status` does not list `config.json` (it is ignored).
- [x] `requirements.txt` has no PyQt entries; build excludes PyQt5/PyQt6/PySide2.
- [x] Version resource describes InterviewCopilot (no system-binary impersonation).
- [ ] Before any release: `git check-ignore -v config.json` prints `.gitignore:20:config.json`;
  search the repo for `AIza` to confirm no key is committed:
  `git grep -n "AIza" -- . ':!architecture.html' ':!DEPLOYMENT.md' || echo CLEAN`

## 8. Troubleshooting quick table

| Symptom | Likely cause | Fix |
|---|---|---|
| `RESULT: FAIL` (diag), hello unanswered | Bad/expired key, no net, TEXT-path quirk | Re-paste key; `python -c "import google.genai"`; try a fallback model |
| `No system-audio (loopback) device` | Output device inactive / `soundcard` broken | Activate output device; `pip install soundcard`; Device Manager → sound |
| HUD shows no frames after 12 s | Silence or wrong loopback device | Play audio; pick the device in Audio tab; lower sensitivity |
| Blank window / missing module in exe | Wrong cwd / partial copy (onedir) | Run from inside `dist\InterviewCopilot\`; ship the whole folder |
| Frozen `--diag` shows nothing | Windowed build has no console | Rebuild with `--console` or run `python main.py --diag` from source |
