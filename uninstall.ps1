# Removes the LocalFlow autostart task and stops the running app. Add -Purge to also delete venv/models/bin
# and the user data in %APPDATA%\LocalFlow.
param([switch]$Purge)

$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
           ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    $argList = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$PSCommandPath`"")
    if ($Purge) { $argList += '-Purge' }
    $p = Start-Process powershell -Verb RunAs -ArgumentList $argList -Wait -PassThru
    exit $p.ExitCode
}

Stop-ScheduledTask -TaskName 'LocalFlow' -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName 'LocalFlow' -Confirm:$false -ErrorAction SilentlyContinue
Get-CimInstance Win32_Process | Where-Object {
    ($_.Name -match 'pythonw?\.exe' -and $_.CommandLine -match '-m localflow') -or
    ($_.Name -eq 'llama-server.exe' -and $_.ExecutablePath -like "$root*")
} | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Write-Host "Autostart task removed and LocalFlow stopped."

if ($Purge) {
    Remove-Item -Recurse -Force (Join-Path $root '.venv'), (Join-Path $root 'models'), (Join-Path $root 'bin') -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force (Join-Path $env:APPDATA 'LocalFlow') -ErrorAction SilentlyContinue
    Write-Host "Purged venv, models, bin and %APPDATA%\LocalFlow."
}
