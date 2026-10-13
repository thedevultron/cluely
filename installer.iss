; ============================================================================
; InterviewCopilot — Inno Setup 6 script (NOT compiled here)
;
; HOW TO BUILD THE INSTALLER:
;   1. Build the app first:  python build_exe.py        (onedir, fast startup)
;      or single file:       python build_exe.py --onefile
;      Output lands in:      dist\InterviewCopilot\
;   2. Install Inno Setup 6: https://jrsoftware.org/isdl.php
;   3. Open this file (installer.iss) in Inno Setup 6 and press Compile (Ctrl+F9).
;   4. The setup exe appears at: dist\installer\InterviewCopilot-Setup-1.0.0.exe
; ============================================================================

#define MyAppName "InterviewCopilot"
#define MyAppVersion "1.0.0"
#define MyAppExe "InterviewCopilot.exe"

[Setup]
AppId={{3A7B2C11-9E4A-4F2B-8C3D-InterviewCopilot1}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
DefaultDirName={autopf}\InterviewCopilot
DefaultGroupName={#MyAppName}
OutputDir=dist\installer
OutputBaseFilename=InterviewCopilot-Setup-{#MyAppVersion}
ArchitecturesAllowed=x64compatible
PrivilegesRequired=lowest
WizardStyle=modern
Compression=lzma2/max
SolidCompression=yes
UninstallDisplayName={#MyAppName}
#if FileExists("app_icon.ico")
SetupIconFile=app_icon.ico
#endif

[Files]
Source: "dist\InterviewCopilot\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExe}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Run]
Filename: "{app}\{#MyAppExe}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent
