[Setup]
AppId={{C1F5E7F5-77AA-4F63-8A95-5D642E4FA9D8}
AppName=LensDiff
AppVersion=0.2.8
AppVerName=LensDiff OFX v0.2.8
AppPublisher=Moaz ELgabry
AppPublisherURL=https://moazelgabry.com
AppSupportURL=https://github.com/MoazElgabry/ME_OFX/issues
AppUpdatesURL=https://github.com/MoazElgabry/ME_OFX
DefaultDirName={commoncf}\OFX\Plugins
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DisableDirPage=yes
DisableProgramGroupPage=yes
OutputDir=.
OutputBaseFilename=LensDiff_v0.2.8_Windows_Installer
Compression=lzma
SolidCompression=yes
PrivilegesRequired=admin

[Files]
Source: "..\bundle\LensDiff.ofx.bundle\*"; DestDir: "{commoncf64}\OFX\Plugins\LensDiff.ofx.bundle"; Flags: ignoreversion recursesubdirs createallsubdirs

[Code]
function ResolveRunning: Boolean;
var
  ResultCode: Integer;
begin
  Result := False;

  if Exec(
      ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'),
      '-NoProfile -ExecutionPolicy Bypass -Command "if (Get-Process Resolve -ErrorAction SilentlyContinue) { exit 1 } else { exit 0 }"',
      '',
      SW_HIDE,
      ewWaitUntilTerminated,
      ResultCode) then
  begin
    Result := (ResultCode = 1);
  end;
end;

function InitializeSetup(): Boolean;
var
  Clicked: Integer;
begin
  while ResolveRunning() do
  begin
    Clicked := SuppressibleMsgBox(
      'DaVinci Resolve is currently running.' + #13#10 +
      'Please close Resolve before installing LensDiff.',
      mbError,
      MB_RETRYCANCEL,
      IDRETRY);

    if Clicked = IDCANCEL then
    begin
      Result := False;
      Exit;
    end;
  end;

  Result := True;
end;
