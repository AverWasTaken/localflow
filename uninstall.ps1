# Stops this copy of LocalFlow and removes the "LocalFlow" startup task if it belongs to this copy (a task that
# runs a different copy is left alone). Other LocalFlow copies keep running.
#   -KeepTask  only stop the app, leave the task registered
#   -Purge     also delete .venv, models, bin, python and the user data in %APPDATA%\LocalFlow
# Exit codes: 0 ok, 2 this copy is still running, 4 its startup task could not be removed.
param([switch]$Purge, [switch]$KeepTask, [switch]$Elevated)

$ErrorActionPreference = 'Continue'
$root = $PSScriptRoot
. (Join-Path $root 'scripts\lfcommon.ps1')

& (Join-Path $root 'scripts\stop.ps1') -Root $root
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: LocalFlow from $root is still running. Quit it from its tray icon and try again." -ForegroundColor Red
    exit 2
}

if (-not $KeepTask) {
    $task = Get-LFTask
    if (($null -eq $task.Exists) -or ($task.Exists -and -not $task.Exe)) {
        # can't tell whether the task is ours: check again with admin rights, or give up
        if ($Elevated -or (Test-LFAdmin)) {
            Write-Host "ERROR: could not check the '$LFTaskName' startup task." -ForegroundColor Red
            exit 4
        }
        Write-Host "Asking for admin to check the startup task..."
        $argList = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$PSCommandPath`"", '-Elevated')
        if ($Purge) { $argList += '-Purge' }
        try {
            $p = Start-Process powershell -Verb RunAs -WindowStyle Hidden -ArgumentList $argList -Wait -PassThru
            exit $p.ExitCode
        } catch {
            Write-Host "ERROR: elevation was declined, so the startup task could not be checked." -ForegroundColor Red
            exit 4
        }
    }
    if ($task.Exists -and (Test-LFUnderRoot $task.Exe $root)) {
        & schtasks.exe /Delete /TN $LFTaskName /F 2>&1 | Out-Null
        if ((Get-LFTask).Exists -ne $false -and -not (Test-LFAdmin)) {
            Write-Host "Asking for admin to remove the startup task..."
            try {
                Start-Process schtasks.exe -Verb RunAs -WindowStyle Hidden -Wait `
                    -ArgumentList @('/Delete', '/TN', $LFTaskName, '/F') | Out-Null
            } catch { Write-Host "Elevation was declined: $($_.Exception.Message)" }
        }
        if ((Get-LFTask).Exists -ne $false) {
            Write-Host "ERROR: could not remove the '$LFTaskName' startup task." -ForegroundColor Red
            exit 4
        }
        Write-Host "Startup task removed."
    } elseif ($task.Exists) {
        Write-Host "The '$LFTaskName' task runs another copy ($($task.Exe)); left it alone."
    }
}
Write-Host "LocalFlow ($root) stopped."

if ($Purge) {
    foreach ($d in '.venv', 'models', 'bin', 'python', '.uv-cache') {
        Remove-Item -Recurse -Force (Join-Path $root $d) -ErrorAction SilentlyContinue
    }
    Remove-Item -Recurse -Force (Join-Path $env:APPDATA 'LocalFlow') -ErrorAction SilentlyContinue
    Write-Host "Purged .venv, models, bin, python and %APPDATA%\LocalFlow."
}
exit 0
