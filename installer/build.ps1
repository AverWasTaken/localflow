# Builds dist\LocalFlow-Setup-<version>.exe with Inno Setup 6.7 or newer (winget install -e --id JRSoftware.InnoSetup).
# The version comes from localflow\__init__.py unless -Version is given.
param([string]$Version)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot

if (-not $Version) {
    $init = Get-Content -LiteralPath (Join-Path $repo 'localflow\__init__.py') -Raw
    if ($init -notmatch '__version__\s*=\s*"([^"]+)"') { throw 'Could not read __version__ from localflow\__init__.py' }
    $Version = $Matches[1]
}
if ($Version -notmatch '^\d+\.\d+\.\d+$') { throw "Version '$Version' is not in x.y.z form." }

$iscc = @(
    (Get-Command ISCC.exe -ErrorAction SilentlyContinue).Source,
    (Join-Path $env:LOCALAPPDATA 'Programs\Inno Setup 6\ISCC.exe'),
    (Join-Path ${env:ProgramFiles(x86)} 'Inno Setup 6\ISCC.exe'),
    (Join-Path $env:ProgramFiles 'Inno Setup 6\ISCC.exe')
) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
if (-not $iscc) { throw 'ISCC.exe (Inno Setup 6.7 or newer) not found. Install it: winget install -e --id JRSoftware.InnoSetup' }

# Windows PowerShell 5.1 reads BOM-less scripts as ANSI, so the shipped scripts must be plain ASCII
$shipped = @(Get-ChildItem -LiteralPath (Join-Path $repo 'scripts') -Filter *.ps1) +
           @(Get-Item (Join-Path $repo 'install.ps1'), (Join-Path $repo 'uninstall.ps1'), (Join-Path $PSScriptRoot 'LocalFlow.iss'))
foreach ($f in $shipped) {
    $bytes = [IO.File]::ReadAllBytes($f.FullName)
    if ($bytes | Where-Object { $_ -gt 127 } | Select-Object -First 1) { throw "$($f.Name) contains non-ASCII characters." }
}

Write-Host "Building LocalFlow $Version with $iscc"
& $iscc "/DAppVersion=$Version" (Join-Path $PSScriptRoot 'LocalFlow.iss')
if ($LASTEXITCODE -ne 0) { throw "ISCC failed (exit code $LASTEXITCODE)" }
$exe = Join-Path $repo "dist\LocalFlow-Setup-$Version.exe"
if (-not (Test-Path $exe)) { throw "Expected $exe after the build." }
Write-Host "Built $exe ($([math]::Round((Get-Item $exe).Length / 1MB, 2)) MB)"
