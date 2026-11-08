# Stops the LocalFlow copy installed in -Root (default: the folder above this script) and nothing else:
# python/pythonw/llama-server processes whose program lives inside -Root, plus their child processes (the venv
# launcher starts the base interpreter as a child, which starts llama-server). Ends the startup task first when
# that task belongs to this copy, so Task Scheduler does not restart the app. Asks for admin only when needed:
# the startup task runs LocalFlow elevated, and without admin an elevated process's program path (and sometimes
# the task itself) can't be read.
# Exit codes: 0 nothing of this copy is running any more, 2 something is still running or can't be checked.
param([string]$Root = (Split-Path -Parent $PSScriptRoot), [switch]$NoElevate)

$ErrorActionPreference = 'Continue'
. (Join-Path $PSScriptRoot 'lfcommon.ps1')
$Root = Get-LFFullPath $Root

# The program a command line starts: the first token, quoted or not.
function Get-ProgramFromCommandLine([string]$CommandLine) {
    $cl = "$CommandLine".TrimStart()
    if (-not $cl) { return $null }
    if ($cl.StartsWith('"')) { $end = $cl.IndexOf('"', 1); if ($end -gt 1) { return $cl.Substring(1, $end - 1) }; return $null }
    $sp = $cl.IndexOf(' ')
    if ($sp -lt 0) { return $cl }
    return $cl.Substring(0, $sp)
}

function Get-OurProcesses {
    $all = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
             Where-Object { $_.ProcessId -ne $PID -and $_.Name -match '^(pythonw?|llama-server)\.exe$' })
    $ids = @{}
    foreach ($p in $all) {
        if ((Test-LFUnderRoot $p.ExecutablePath $Root) -or
            (Test-LFUnderRoot (Get-ProgramFromCommandLine $p.CommandLine) $Root)) { $ids[[int]$p.ProcessId] = $p }
    }
    $added = $true
    while ($added) {   # children (and grandchildren) of matched processes
        $added = $false
        foreach ($p in $all) {
            if (-not $ids.ContainsKey([int]$p.ProcessId) -and $ids.ContainsKey([int]$p.ParentProcessId)) {
                $ids[[int]$p.ProcessId] = $p; $added = $true
            }
        }
    }
    return @($ids.Values)
}

# 'running' / 'idle' = the task belongs to this copy; 'other' = no task, or it runs another copy;
# 'unknown' = the task (or its state) can't be read
function Get-OurTaskState {
    $t = Get-LFTask
    if ($t.Exists -eq $false) { return 'other' }
    if (-not $t.Exists -or -not $t.Exe) { return 'unknown' }
    if (-not (Test-LFUnderRoot $t.Exe $Root)) { return 'other' }
    if ($null -eq $t.Running) { return 'unknown' }
    if ($t.Running) { return 'running' }
    return 'idle'
}

function Stop-Ours {
    if ((Get-OurTaskState) -eq 'running') {
        Write-Host "Ending the '$LFTaskName' task"
        & schtasks.exe /End /TN $LFTaskName 2>&1 | Out-Null
        Start-Sleep -Milliseconds 500
    }
    foreach ($p in Get-OurProcesses) {
        Write-Host "Stopping $($p.Name) (pid $($p.ProcessId))"
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Milliseconds 800
}

function Get-Blockers {
    $b = @(Get-OurProcesses | ForEach-Object { "$($_.Name) ($($_.ProcessId))" })
    switch (Get-OurTaskState) {
        'running' { $b += "the '$LFTaskName' task" }
        'unknown' { $b += "the '$LFTaskName' task (could not be checked)" }
    }
    return ,$b
}

Stop-Ours
$blockers = Get-Blockers

if ($blockers.Count -gt 0 -and -not $NoElevate -and -not (Test-LFAdmin)) {
    Write-Host "Asking for admin to stop LocalFlow ($($blockers -join ', '))..."
    $childExit = 1
    try {
        $argList = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$PSCommandPath`"",
                     '-Root', "`"$Root`"", '-NoElevate')
        $p = Start-Process powershell -Verb RunAs -WindowStyle Hidden -ArgumentList $argList -Wait -PassThru
        $childExit = $p.ExitCode
    } catch { Write-Host "Elevation was declined: $($_.Exception.Message)" }
    Start-Sleep -Milliseconds 500
    # the elevated run is the authority on what this (unelevated) run can't see; still check what we can see
    $visible = @(Get-OurProcesses)
    if ($childExit -eq 0 -and $visible.Count -eq 0 -and (Get-OurTaskState) -ne 'running') { exit 0 }
    $blockers = Get-Blockers
    if ($blockers.Count -eq 0) { $blockers = @('see the elevated run') }
}

if ($blockers.Count -gt 0) {
    Write-Host "Still running or not checkable: $($blockers -join ', ')"
    exit 2
}
exit 0
