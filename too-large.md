# Files Excluded from Git (Size / Artifact Reasons)

| File name | Size (bytes) | Size (MB) | Full path | Importance |
|---|---|---|---|---|
| InterviewCopilot-Setup-1.0.0.exe | 82,572,422 | 78.75 MB | C:\Users\Amir\cluely\dist\installer\InterviewCopilot-Setup-1.0.0.exe | Build artifact: Windows installer produced by Inno Setup (`installer.iss`) during the PyInstaller packaging step. Fully regenerable via `build_exe.py` (plus Inno Setup). The repo builds fine without it. |

## Also excluded (artifact/log/secret categories, all regenerable)

| Path | Reason | Regenerable via |
|---|---|---|
| build/ | PyInstaller intermediate build dir | build_exe.py |
| dist/ | PyInstaller output dir (incl. the 78.75 MB installer above) | build_exe.py |
| __pycache__/ | Python bytecode cache | automatic on next run |
| app.log | runtime log | app run |
| config.json | contains live API key — excluded for security, NOT size | first-run config creation / manual edit |

Note: `transcripts/` was reviewed — files are small markdown session notes (total ~5.9 KB), so they are tracked, not ignored.
