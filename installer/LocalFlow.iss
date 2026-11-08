; LocalFlow setup program (Inno Setup 6.7 or newer). Build with installer\build.ps1, which passes /DAppVersion=x.y.z.
; The setup is small: it copies the LocalFlow source and then runs install.ps1, which downloads Python,
; the Python packages, llama.cpp and the models into the install folder.
; Extra setup switches (besides the standard /SILENT, /VERYSILENT, /DIR=...):
;   /SKIPMODEL  don't download the AI cleanup model
;   /NOTASK     don't register the "LocalFlow" startup task (no admin prompt)
; Setup exit code 10 = files were copied but install.ps1 failed (see install.log in the install folder).
; Keep this file ASCII only.

#if Ver < EncodeVer(6, 7, 0)
  #error Inno Setup 6.7 or newer is required (the RedirectionGuard directive below).
#endif

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{DEA69D86-4F94-4CB2-AD7C-3B8DF994BBF5}
AppName=LocalFlow
AppVersion={#AppVersion}
AppVerName=LocalFlow {#AppVersion}
AppPublisher=AverWasTaken
AppPublisherURL=https://github.com/AverWasTaken/localflow
AppSupportURL=https://github.com/AverWasTaken/localflow/issues
AppUpdatesURL=https://github.com/AverWasTaken/localflow/releases
VersionInfoVersion={#AppVersion}
DefaultDirName={localappdata}\Programs\LocalFlow
DefaultGroupName=LocalFlow
DisableProgramGroupPage=yes
DisableWelcomePage=no
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
ExtraDiskSpaceRequired=10737418240
OutputDir=..\dist
OutputBaseFilename=LocalFlow-Setup-{#AppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
CloseApplications=no
RestartApplications=no
SetupLogging=yes
UninstallDisplayName=LocalFlow
UsePreviousTasks=yes
; Inno Setup 6.7 turns RedirectionGuard on by default and child processes inherit it. uv (run by install.ps1)
; makes a junction python\cpython-3.12-windows-x86_64-none; made by a non-admin process it is "untrusted", and
; with the guard on, uv and Python then fail with error 448 (untrusted mount point). The guard is meant for
; setups that run elevated; this one runs as the user (PrivilegesRequired=lowest).
RedirectionGuard=no

[Messages]
WelcomeLabel2=This will install [name/ver] on your computer.%n%nAfter copying its files, setup downloads Python, the Python packages, llama.cpp and the AI models (about 7 GB in total, or about 2.5 GB without the AI cleanup model). Keep the computer online until it finishes. A PowerShell window shows the progress.%n%nWindows asks for admin permission once, to add the startup task that lets LocalFlow type into apps running as administrator.

[Tasks]
Name: "downloadmodel"; Description: "Download the AI cleanup model (about 4.5 GB, recommended with an NVIDIA GPU)"; GroupDescription: "Optional downloads:"

[InstallDelete]
; no stale modules left behind by an older version
Type: filesandordirs; Name: "{app}\localflow"

[Dirs]
Name: "{app}\localflow\assets"

[Files]
Source: "..\localflow\*.py"; DestDir: "{app}\localflow"; Flags: ignoreversion
Source: "..\localflow\assets\*"; DestDir: "{app}\localflow\assets"; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist
Source: "..\scripts\*.ps1"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "..\scripts\launch.pyw"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "..\install.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\uninstall.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\requirements.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion isreadme

[Icons]
Name: "{group}\LocalFlow"; Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: """{app}\scripts\launch.pyw"""; WorkingDir: "{app}"; Comment: "Local push-to-talk dictation"
Name: "{group}\Uninstall LocalFlow"; Filename: "{uninstallexe}"

[Run]
Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: """{app}\scripts\launch.pyw"""; WorkingDir: "{app}"; Description: "Launch LocalFlow"; Flags: postinstall nowait skipifsilent; Check: SetupSucceeded

[UninstallDelete]
; (before any of this, CurUninstallStepChanged runs uninstall.ps1 to stop this copy and remove its startup task)
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\bin"
Type: filesandordirs; Name: "{app}\models"
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\.uv-cache"
Type: filesandordirs; Name: "{app}\localflow"
Type: filesandordirs; Name: "{app}\scripts"
Type: files; Name: "{app}\install.log"
Type: dirifempty; Name: "{app}"

[Code]
var
  SetupFailed: Boolean;
  DeleteUserData: Boolean;

function HasSwitch(const Name: String): Boolean;
var
  I: Integer;
begin
  Result := False;
  for I := 1 to ParamCount do
    if CompareText(ParamStr(I), Name) = 0 then
    begin
      Result := True;
      Exit;
    end;
end;

function PowerShellExe: String;
begin
  Result := ExpandConstant('{sysnative}\WindowsPowerShell\v1.0\powershell.exe');
end;

function SetupSucceeded: Boolean;
begin
  Result := not SetupFailed;
end;

function GetCustomSetupExitCode: Integer;
begin
  if SetupFailed then
    Result := 10
  else
    Result := 0;
end;

{ A folder setup must not install into: a LocalFlow source checkout (git clone or unzipped source), whose
  localflow folder the InstallDelete section would wipe and the uninstaller would delete. Returns '' when fine. }
function BadInstallDir(const Dir: String): String;
var
  D: String;
begin
  Result := '';
  D := AddBackslash(Dir);
  if DirExists(D + '.git') or (FileExists(D + 'install.ps1') and not FileExists(D + 'unins000.dat')) then
    Result := Dir + ' looks like a LocalFlow source folder (not one made by this setup). ' +
              'Pick another folder, for example ' + ExpandConstant('{localappdata}\Programs\LocalFlow') + '.';
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  Err: String;
begin
  Result := True;
  if CurPageID = wpSelectDir then
  begin
    Err := BadInstallDir(RemoveBackslashUnlessRoot(WizardDirValue));
    if Err <> '' then
    begin
      MsgBox(Err, mbError, MB_OK);
      Result := False;
    end;
  end;
end;

{ Upgrade / reinstall: stop the copy running from this folder (and only that copy) before files are replaced. }
function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  Dir, Params: String;
  RC: Integer;
begin
  Dir := RemoveBackslashUnlessRoot(WizardDirValue);
  Result := BadInstallDir(Dir);   { also covers silent installs with /DIR= }
  if Result <> '' then
    Exit;
  if not FileExists(AddBackslash(Dir) + 'install.ps1') then
    Exit;
  ExtractTemporaryFile('stop.ps1');
  ExtractTemporaryFile('lfcommon.ps1');
  Params := '-NoProfile -ExecutionPolicy Bypass -File "' + ExpandConstant('{tmp}\stop.ps1') + '" -Root "' + Dir + '"';
  if not Exec(PowerShellExe, Params, '', SW_HIDE, ewWaitUntilTerminated, RC) then
    RC := -1;
  if RC <> 0 then
    Result := 'LocalFlow is still running from ' + Dir + '. Quit it from its tray icon, then run setup again.';
end;

procedure RunInstallScript;
var
  Params, Msg: String;
  RC, Show: Integer;
  Again: Boolean;
begin
  Params := '-NoProfile -ExecutionPolicy Bypass -File "' + ExpandConstant('{app}\install.ps1') + '" -NoStart';
  if not WizardSilent then
    Params := Params + ' -PauseOnError';
  if HasSwitch('/SKIPMODEL') or not WizardIsTaskSelected('downloadmodel') then
    Params := Params + ' -SkipModel';
  if HasSwitch('/NOTASK') then
    Params := Params + ' -NoTask';
  if WizardSilent then
    Show := SW_HIDE
  else
    Show := SW_SHOW;
  Log('Running install.ps1: ' + Params);
  repeat
    WizardForm.StatusLabel.Caption := 'Downloading and setting up LocalFlow (progress is in the PowerShell window)...';
    WizardForm.FilenameLabel.Caption := 'Python, packages, llama.cpp and models. This can take a while.';
    if not Exec(PowerShellExe, Params, ExpandConstant('{app}'), Show, ewWaitUntilTerminated, RC) then
      RC := -1;
    Log('install.ps1 exit code: ' + IntToStr(RC));
    SetupFailed := RC <> 0;
    Again := False;
    if SetupFailed then
    begin
      Msg := 'LocalFlow setup did not finish (code ' + IntToStr(RC) + ').' + #13#10 + #13#10 +
             'The details are in:' + #13#10 + ExpandConstant('{app}\install.log') + #13#10 + #13#10 +
             'Click Retry to try again, or Cancel to leave it as it is. Running setup again later picks up where it stopped.';
      Again := SuppressibleMsgBox(Msg, mbError, MB_RETRYCANCEL, IDCANCEL) = IDRETRY;
    end;
  until not Again;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
    RunInstallScript;
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  if (CurPageID = wpFinished) and SetupFailed then
    WizardForm.FinishedLabel.Caption :=
      'LocalFlow was copied to your computer, but downloading its parts did not finish.' + #13#10 + #13#10 +
      'Run this setup again to finish (it picks up where it stopped). The log is in ' +
      ExpandConstant('{app}\install.log') + '.';
end;

{ Stops this copy and removes its startup task. Returns False if that did not work and the user gave up
  (or the uninstall is silent). }
function StopBeforeUninstall: Boolean;
var
  Params, Msg: String;
  RC: Integer;
begin
  Result := True;
  if not FileExists(ExpandConstant('{app}\uninstall.ps1')) then
    Exit;
  Params := '-NoProfile -ExecutionPolicy Bypass -File "' + ExpandConstant('{app}\uninstall.ps1') + '"';
  repeat
    if not Exec(PowerShellExe, Params, ExpandConstant('{app}'), SW_HIDE, ewWaitUntilTerminated, RC) then
      RC := -1;
    Log('uninstall.ps1 exit code: ' + IntToStr(RC));
    Result := RC = 0;
    if Result or UninstallSilent then
      Exit;
    if RC = 4 then
      Msg := 'The LocalFlow startup task could not be removed (the admin prompt may have been declined).'
    else
      Msg := 'LocalFlow is still running and could not be stopped.';
    if MsgBox(Msg + #13#10 + #13#10 + 'Quit LocalFlow from its tray icon, then click Retry. ' +
              'Cancel stops the uninstall and leaves LocalFlow installed.', mbError, MB_RETRYCANCEL) <> IDRETRY then
      Exit;
  until False;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Hub: String;
begin
  if CurUninstallStep = usUninstall then
  begin
    if not StopBeforeUninstall then
      Abort;
    DeleteUserData := False;
    if not UninstallSilent then
      DeleteUserData := MsgBox('Also delete your LocalFlow settings, history, meeting transcripts and recordings ' +
        '(' + ExpandConstant('{userappdata}\LocalFlow') + ') and the downloaded speech models?' + #13#10 + #13#10 +
        'Click No to keep them, for example if you plan to install LocalFlow again.',
        mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES;
  end;
  if (CurUninstallStep = usPostUninstall) and DeleteUserData then
  begin
    DelTree(ExpandConstant('{userappdata}\LocalFlow'), True, True, True);
    Hub := GetEnv('USERPROFILE') + '\.cache\huggingface\hub';
    DelTree(Hub + '\models--h2oai--faster-whisper-large-v3-turbo', True, True, True);
    DelTree(Hub + '\models--Systran--faster-whisper-small.en', True, True, True);
  end;
end;
