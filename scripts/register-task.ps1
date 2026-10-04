# Registers the "LocalFlow" logon task: runs as the current user with highest privileges (no UAC prompt),
# restarts on failure, never times out, keeps running on battery.
param([switch]$Start)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
           ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "Elevating to register the scheduled task..."
    $argList = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$PSCommandPath`"")
    if ($Start) { $argList += '-Start' }
    $p = Start-Process powershell -Verb RunAs -ArgumentList $argList -Wait -PassThru
    exit $p.ExitCode
}

$pyw = Join-Path $root '.venv\Scripts\pythonw.exe'
if (-not (Test-Path $pyw)) { throw "pythonw not found at $pyw - run install.ps1 first" }

$user = "$env:USERDOMAIN\$env:USERNAME"
$action = New-ScheduledTaskAction -Execute $pyw -Argument '-m localflow' -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $user
$trigger.Delay = 'PT5S'
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 99 -RestartInterval (New-TimeSpan -Minutes 1) `
    -StartWhenAvailable -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName 'LocalFlow' -Action $action -Trigger $trigger -Principal $principal `
    -Settings $settings -Description 'LocalFlow local dictation (starts at logon, elevated)' -Force | Out-Null
Write-Host "Scheduled task 'LocalFlow' registered for $user (run level: Highest)."
if ($Start) {
    Start-ScheduledTask -TaskName 'LocalFlow'
    Write-Host "Started."
}
