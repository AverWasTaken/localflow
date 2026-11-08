# Registers the "LocalFlow" logon task: runs as the current user with highest privileges (no UAC prompt),
# restarts on failure, never times out, keeps running on battery.
# Refuses to replace an existing LocalFlow task that runs a different copy of LocalFlow (or one it can't read)
# unless -Force is given. The task name is machine-wide, so there is one LocalFlow startup task per computer.
# Exit codes: 0 ok, 3 the task belongs to another copy / can't be checked, 1 other error.
param([switch]$Start, [switch]$Force, [string]$TaskUser)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'lfcommon.ps1')
$root = Split-Path -Parent $PSScriptRoot
$pyw = Join-Path $root '.venv\Scripts\pythonw.exe'
# the user the task runs as: the person who started this script, even if the admin prompt was answered with
# another account's credentials
if (-not $TaskUser) { $TaskUser = [Security.Principal.WindowsIdentity]::GetCurrent().Name }

function Assert-TaskIsOurs([bool]$Authoritative) {
    if ($Force) { return }
    $t = Get-LFTask
    if ($t.Exists -eq $false) { return }
    if ($t.Exists -and $t.Exe -and (Test-LFUnderRoot $t.Exe $root)) { return }
    if ($null -eq $t.Exists -and -not $Authoritative) { return }   # can't read it yet; checked again once elevated
    if ($t.Exists -and $t.Exe) {
        Write-Host "A '$LFTaskName' startup task already exists and runs another copy of LocalFlow:" -ForegroundColor Yellow
        Write-Host "  $($t.Exe)"
    } else {
        Write-Host "A '$LFTaskName' startup task may already exist but its program could not be read." -ForegroundColor Yellow
    }
    Write-Host "Left it alone. Uninstall that copy first, or re-run this script with -Force to point the task here:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Force"
    exit 3
}

Assert-TaskIsOurs $false    # checked before elevating so the message shows in this window

if (-not (Test-LFAdmin)) {
    Write-Host "Elevating to register the scheduled task..."
    $argList = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$PSCommandPath`"", '-TaskUser', "`"$TaskUser`"")
    if ($Start) { $argList += '-Start' }
    if ($Force) { $argList += '-Force' }
    $p = Start-Process powershell -Verb RunAs -ArgumentList $argList -Wait -PassThru
    exit $p.ExitCode
}

Assert-TaskIsOurs $true    # elevated: the answer is authoritative now
if (-not (Test-Path $pyw)) { throw "pythonw not found at $pyw - run install.ps1 first" }

$action = New-ScheduledTaskAction -Execute $pyw -Argument '-m localflow' -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $TaskUser
$trigger.Delay = 'PT5S'
$principal = New-ScheduledTaskPrincipal -UserId $TaskUser -LogonType Interactive -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 99 -RestartInterval (New-TimeSpan -Minutes 1) `
    -StartWhenAvailable -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $LFTaskName -Action $action -Trigger $trigger -Principal $principal `
    -Settings $settings -Description 'LocalFlow local dictation (starts at logon, elevated)' -Force | Out-Null
Write-Host "Scheduled task '$LFTaskName' registered for $TaskUser (run level: Highest)."
if ($Start) {
    Start-ScheduledTask -TaskName $LFTaskName
    Write-Host "Started."
}
