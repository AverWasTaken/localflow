# Shared helpers for the LocalFlow scripts (dot-source this file). ASCII only: Windows PowerShell 5.1 reads
# BOM-less files as ANSI.

$LFTaskName = 'LocalFlow'

function Test-LFAdmin {
    ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
    ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-LFFullPath([string]$Path) {
    [IO.Path]::GetFullPath($Path.Trim().Trim('"')).TrimEnd('\')
}

# True when $Path is $Root itself or lies inside it (case-insensitive, no prefix false positives).
function Test-LFUnderRoot([string]$Path, [string]$Root) {
    if (-not $Path -or -not $Root) { return $false }
    try { $p = Get-LFFullPath $Path; $r = Get-LFFullPath $Root } catch { return $false }
    return ($p -ieq $r) -or $p.StartsWith($r + '\', [StringComparison]::OrdinalIgnoreCase)
}

# Reads the task through the Task Scheduler COM API (locale independent).
# Returns @{ Exists = $true/$false/$null (null = could not tell); Exe = the first exec action's program or $null;
#            Running = $true when an instance is running }.
function Get-LFTask {
    try {
        $svc = New-Object -ComObject Schedule.Service
        $svc.Connect()
        $task = $svc.GetFolder('\').GetTask($LFTaskName)
    } catch {
        $hr = $_.Exception.HResult
        if ($_.Exception.InnerException) { $hr = $_.Exception.InnerException.HResult }
        if ($hr -eq -2147024894 -or $hr -eq -2147024893) {   # 0x80070002 / 0x80070003: no such task
            return @{ Exists = $false; Exe = $null; Running = $false }
        }
        return @{ Exists = $null; Exe = $null; Running = $null }
    }
    $exe = $null
    try {
        foreach ($a in $task.Definition.Actions) {
            if ($a.Type -eq 0 -and $a.Path) { $exe = [Environment]::ExpandEnvironmentVariables($a.Path.Trim().Trim('"')); break }
        }
    } catch {}
    $running = $null
    try { $running = ($task.State -eq 4) } catch {}
    return @{ Exists = $true; Exe = $exe; Running = $running }
}
