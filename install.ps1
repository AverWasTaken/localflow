# LocalFlow installer: Python + venv + deps + llama.cpp server + cleanup model + speech model + autostart task.
# The Windows setup program (LocalFlow-Setup-x.y.z.exe) runs this script; you can also run it by hand from a
# normal PowerShell:  powershell -ExecutionPolicy Bypass -File install.ps1
# Safe to run again: it repairs whatever is missing and skips what is already in place.
# Flags:
#   -NoStart       don't launch LocalFlow at the end
#   -SkipModel     don't download the AI cleanup model (about 4.5 GB)
#   -NoTask        don't register the "LocalFlow" startup task (no admin prompt)
#   -PauseOnError  on failure, wait for Enter before the window closes
# Everything is logged to install.log next to this script. Exit code 0 = success.
param([switch]$NoStart, [switch]$SkipModel, [switch]$NoTask, [switch]$PauseOnError)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # the progress bar makes Invoke-WebRequest very slow in Windows PowerShell
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
$root = $PSScriptRoot
Set-Location -LiteralPath $root
. (Join-Path $root 'scripts\lfcommon.ps1')

# ---------------------------------------------------------------- pinned downloads (sha256 from the release pages)
$UvVersion = '0.13.7'
$UvAsset   = 'uv-x86_64-pc-windows-msvc.zip'
$UvSha256  = '0cdc5b38fe5a8168ed6a25814f5ba43c21d313dd3187314fccb37b7cec46e021'
$LlamaTag  = 'b12211'
$LlamaAssets = [ordered]@{
    'llama-b12211-bin-win-cuda-13.5-x64.zip' = 'dc396a25a42729178d991d3913e6fd3619cfb446f6af0937bdec5d03e5375b56'
    'cudart-llama-bin-win-cuda-13.5-x64.zip' = 'c2bf4cc202bfbaaa7a0b75f5082457c3811326947bb216059b72839d844e9cba'
}
$GgufRepo = 'unsloth/Qwen3.5-4B-GGUF'
$GgufFile = 'Qwen3.5-4B-Q8_0.gguf'
$WhisperGpu = 'h2oai/faster-whisper-large-v3-turbo'
$WhisperCpu = 'Systran/faster-whisper-small.en'
if ($env:LOCALFLOW_TEST_BAD_SHA256 -eq '1') {   # test hook: makes every checksum check fail
    $UvSha256 = '0' * 64
    foreach ($k in @($LlamaAssets.Keys)) { $LlamaAssets[$k] = '0' * 64 }
}

$log = Join-Path $root 'install.log'
$binDir = Join-Path $root 'bin'
$uvCache = Join-Path $root '.uv-cache'

function Step($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Warn($m) { Write-Host "WARNING: $m" -ForegroundColor Yellow }

# Runs a program, copies its output (stdout + stderr) into the console and the log, throws on a non-zero exit.
function Invoke-Logged([string]$Exe, [string[]]$Arguments, [string]$What) {
    $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try {
        & $Exe @Arguments 2>&1 | ForEach-Object { Write-Host "$_" }
        $code = $LASTEXITCODE
    } finally { $ErrorActionPreference = $prev }
    if ($code -ne 0) { throw "$What failed (exit code $code). See the messages above." }
}

# Same, but leaves the output alone so progress bars work (used for the big downloads).
function Invoke-Direct([string]$Exe, [string[]]$Arguments, [string]$What) {
    $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try {
        & $Exe @Arguments
        $code = $LASTEXITCODE
    } finally { $ErrorActionPreference = $prev }
    if ($code -ne 0) { throw "$What failed (exit code $code)." }
}

# Downloads to <dest>.part, checks the sha256, then renames to <dest>.
function Get-Verified([string]$Url, [string]$Dest, [string]$Sha256) {
    $part = "$Dest.part"
    $name = [IO.Path]::GetFileName($Dest)
    $curl = Join-Path $env:SystemRoot 'System32\curl.exe'
    $ok = $false
    for ($i = 1; $i -le 3 -and -not $ok; $i++) {
        if (Test-Path -LiteralPath $part) { Remove-Item -LiteralPath $part -Force }
        try {
            Write-Host "Downloading $name"
            if (Test-Path $curl) {
                Invoke-Direct $curl @('-L', '--fail', '--retry', '3', '--retry-delay', '2', '--progress-bar', '-o', $part, $Url) 'curl'
            } else {
                Invoke-WebRequest -Uri $Url -OutFile $part -UseBasicParsing
            }
            $ok = $true
        } catch {
            Warn "download attempt $i of 3 failed: $($_.Exception.Message)"
            if ($i -lt 3) { Start-Sleep -Seconds 3 }
        }
    }
    if (-not $ok) { throw "Could not download $Url. Check your internet connection and run setup again." }
    $hash = (Get-FileHash -LiteralPath $part -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($hash -ne $Sha256.ToLowerInvariant()) {
        Remove-Item -LiteralPath $part -Force
        throw "Checksum mismatch for $name (expected $Sha256, got $hash). The download was damaged or changed upstream; run setup again."
    }
    Move-Item -LiteralPath $part -Destination $Dest -Force
}

function Expand-Zip([string]$Zip, [string]$Dest) {
    New-Item -ItemType Directory -Force $Dest | Out-Null
    $tar = Join-Path $env:SystemRoot 'System32\tar.exe'
    if (Test-Path $tar) { Invoke-Logged $tar @('-xf', $Zip, '-C', $Dest) "Unpacking $([IO.Path]::GetFileName($Zip))" }
    else { Expand-Archive -LiteralPath $Zip -DestinationPath $Dest -Force }
}

function Get-FreeGB([string]$Path) {
    try { [math]::Round((New-Object IO.DriveInfo ([IO.Path]::GetPathRoot($Path))).AvailableFreeSpace / 1GB, 1) }
    catch { return [double]::MaxValue }
}

# Build tag of an installed llama-server ("b12211"), from VERSION or from `llama-server --version`.
function Get-LlamaBuild([string]$Dir) {
    $verFile = Join-Path $Dir 'VERSION'
    $exe = Join-Path $Dir 'llama-server.exe'
    if (-not (Test-Path $exe)) { return '' }
    if (Test-Path $verFile) { return (Get-Content -LiteralPath $verFile -Raw).Trim() }
    $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try { $out = (& $exe --version 2>&1 | ForEach-Object { "$_" }) -join "`n" } catch { $out = '' }
    finally { $ErrorActionPreference = $prev }
    # newer builds print "version: 0.9.0-dev (build 12211, ...)", older ones "version: 11381 (...)"
    if ($out -match '\(build (\d+)') { return "b$($Matches[1])" }
    if ($out -match 'version:\s*(\d+)\s*\(') { return "b$($Matches[1])" }
    return ''
}

function Install-LocalFlow {
    # ------------------------------------------------------------ prerequisites
    Step 'Checking prerequisites'
    $os = [Environment]::OSVersion.Version
    if ($os.Major -lt 10 -or -not [Environment]::Is64BitOperatingSystem) {
        throw 'LocalFlow needs 64-bit Windows 10 or 11.'
    }
    Write-Host "Windows $($os.Major).$($os.Minor).$($os.Build): ok"
    Write-Host "Install folder: $root"

    $script:hasGpu = $false
    $smi = (Get-Command nvidia-smi -ErrorAction SilentlyContinue).Source
    if (-not $smi -and (Test-Path "$env:SystemRoot\System32\nvidia-smi.exe")) { $smi = "$env:SystemRoot\System32\nvidia-smi.exe" }
    if ($smi) {
        $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
        $info = & $smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>$null
        $smiExit = $LASTEXITCODE
        $ErrorActionPreference = $prev
        if ($smiExit -eq 0 -and $info) {
            $script:hasGpu = $true
            $first = ($info | Select-Object -First 1).Split(',') | ForEach-Object { $_.Trim() }
            Write-Host "GPU: $($first[0]), driver $($first[1]), $($first[2])"
            $drv = 0
            if ([int]::TryParse(($first[1].Split('.')[0]), [ref]$drv) -and $drv -lt 580) {
                Warn "driver $($first[1]) is older than 580. The CUDA 13 llama.cpp build needs driver 580 or newer; update it from nvidia.com/drivers or the cleanup model will fall back to the CPU (or fail to load)."
            }
            $mb = 0
            if ([int]::TryParse(($first[2] -replace '[^\d]', ''), [ref]$mb) -and $mb -lt 5500) {
                Warn "only $($first[2]) of VRAM. large-v3-turbo plus the cleanup model want about 6 GB. Set stt.model to small.en in config.json if loading fails."
            }
        }
    }
    if (-not $script:hasGpu) {
        Warn 'No NVIDIA GPU / driver found (nvidia-smi is missing or failed).'
        Write-Host '  LocalFlow will still run, on the CPU: speech-to-text drops to the small.en model (int8) and'
        Write-Host '  the AI cleanup step is switched off in a fresh config (a 4B model on CPU is too slow to be useful;'
        Write-Host '  a built-in regex cleaner handles fillers and punctuation instead). Dictation will be slower than on a GPU.'
    }

    $venvPy = Join-Path $root '.venv\Scripts\python.exe'
    $gguf = Join-Path $root "models\$GgufFile"
    $llamaDir = Join-Path $binDir 'llama'
    $hfHub = if ($env:HF_HUB_CACHE) { $env:HF_HUB_CACHE } elseif ($env:HF_HOME) { Join-Path $env:HF_HOME 'hub' } `
             else { Join-Path $env:USERPROFILE '.cache\huggingface\hub' }
    $whisperRepo = if ($script:hasGpu) { $WhisperGpu } else { $WhisperCpu }
    $whisperCached = Test-Path (Join-Path $hfHub ('models--' + ($whisperRepo -replace '/', '--')))

    # rough free-space need in GB (downloads + unpacked size), counting only what is still missing
    $need = 0.5
    if (-not (Test-Path $venvPy)) { $need += 4 } else { $need += 1 }
    if ((Get-LlamaBuild $llamaDir) -ne $LlamaTag) { $need += 2 }
    if (-not $SkipModel -and -not (Test-Path $gguf)) { $need += 5 }
    $needHf = if ($whisperCached) { 0 } elseif ($script:hasGpu) { 2 } else { 1 }
    $rootDrive = [IO.Path]::GetPathRoot($root); $hfDrive = [IO.Path]::GetPathRoot($hfHub)
    if ($rootDrive -ieq $hfDrive) { $need += $needHf; $needHf = 0 }
    $free = Get-FreeGB $root
    Write-Host "Free space on ${rootDrive}: $free GB (need about $need GB)"
    if ($free -lt $need) { throw "Not enough free disk space on $rootDrive ($free GB free, about $need GB needed). Free up some space and run setup again." }
    if ($needHf -gt 0) {
        $freeHf = Get-FreeGB $hfHub
        if ($freeHf -lt $needHf) { throw "Not enough free disk space on $hfDrive for the speech model ($freeHf GB free, about $needHf GB needed)." }
    }

    # ------------------------------------------------------------ uv (pinned, private copy)
    $uvDir = Join-Path $binDir 'uv'
    $uv = Join-Path $uvDir 'uv.exe'
    $uvOk = $false
    if (Test-Path $uv) {
        $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
        $uvVer = "$(& $uv --version 2>&1)"
        $ErrorActionPreference = $prev
        $uvOk = $uvVer -match "^uv $([regex]::Escape($UvVersion))( |$)"
    }
    if ($uvOk) { Write-Host "uv $UvVersion`: ok" }
    else {
        Step "Downloading uv $UvVersion (Python package manager)"
        New-Item -ItemType Directory -Force $binDir | Out-Null
        $zip = Join-Path $binDir $UvAsset
        Get-Verified "https://github.com/astral-sh/uv/releases/download/$UvVersion/$UvAsset" $zip $UvSha256
        $tmp = Join-Path $binDir 'uv.new'
        if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
        Expand-Zip $zip $tmp
        Remove-Item -LiteralPath $zip -Force
        $found = Get-ChildItem -LiteralPath $tmp -Recurse -Filter 'uv.exe' | Select-Object -First 1
        if (-not $found) { throw 'uv.exe was not found in the downloaded archive.' }
        if (Test-Path $uvDir) { Remove-Item -Recurse -Force $uvDir }
        New-Item -ItemType Directory -Force $uvDir | Out-Null
        Get-ChildItem -LiteralPath $found.DirectoryName -File | Move-Item -Destination $uvDir -Force
        Remove-Item -Recurse -Force $tmp
    }

    # keep uv's Python, cache and settings inside the install folder
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $root 'python'
    $env:UV_PYTHON_PREFERENCE = 'only-managed'
    $env:UV_PYTHON_DOWNLOADS = 'automatic'
    $env:UV_PYTHON_INSTALL_BIN = '0'
    $env:UV_PYTHON_INSTALL_REGISTRY = '0'
    $env:UV_CACHE_DIR = $uvCache
    $env:UV_NO_PROGRESS = '1'

    # ------------------------------------------------------------ python env
    # uv keeps a link (junction) python\cpython-3.12-windows-x86_64-none -> cpython-3.12.<n>-windows-x86_64-none
    # and the venv's pyvenv.cfg points at it. Clear out what an interrupted run may have left in its place.
    $pyLink = Join-Path $env:UV_PYTHON_INSTALL_DIR 'cpython-3.12-windows-x86_64-none'
    $linkItem = Get-Item -LiteralPath $pyLink -Force -ErrorAction SilentlyContinue
    if ($linkItem) {
        if (-not ($linkItem.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            if (@(Get-ChildItem -LiteralPath $pyLink -Force).Count -gt 0) {
                throw "$pyLink should be a link made by uv but is a folder with files in it. Delete that folder and run setup again."
            }
            Warn "Removing the empty folder $pyLink (uv makes a link there)."
            [IO.Directory]::Delete($pyLink, $false)
        } else {
            $linkTarget = @($linkItem.Target)[0]
            if ($linkTarget -and -not [IO.Path]::IsPathRooted($linkTarget)) { $linkTarget = Join-Path $env:UV_PYTHON_INSTALL_DIR $linkTarget }
            if (-not $linkTarget -or -not [IO.Directory]::Exists($linkTarget)) {
                Warn "Removing the broken Python link $pyLink (it points to a folder that is gone)."
                [IO.Directory]::Delete($pyLink, $false)   # removes only the link, never what it points to
            }
        }
    }
    Step 'Installing Python 3.12 (private copy; skipped if already there)'
    Invoke-Logged $uv @('python', 'install', '3.12') 'Installing Python 3.12'

    if (Test-Path $venvPy) {
        $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
        & $venvPy -c 'import sys' 2>$null
        $venvOk = ($LASTEXITCODE -eq 0)
        $ErrorActionPreference = $prev
        if (-not $venvOk) {
            Warn 'The existing .venv is broken (its Python is gone); recreating it.'
            Remove-Item -Recurse -Force (Join-Path $root '.venv')
        }
    }
    if (-not (Test-Path $venvPy)) {
        Step 'Creating the Python 3.12 environment'
        Invoke-Logged $uv @('venv', '--python', '3.12', (Join-Path $root '.venv')) 'Creating the Python environment'
    } else { Write-Host 'Python environment: ok' }

    Step 'Installing Python packages (about 3 GB the first time)'
    Invoke-Logged $uv @('pip', 'install', '--python', $venvPy, '-r', (Join-Path $root 'requirements.txt')) 'Installing Python packages'

    # ------------------------------------------------------------ llama.cpp
    $have = Get-LlamaBuild $llamaDir
    if ($have -eq $LlamaTag) {
        Write-Host "llama.cpp $LlamaTag`: ok"
        $verFile = Join-Path $llamaDir 'VERSION'
        if (-not (Test-Path $verFile)) { [IO.File]::WriteAllText($verFile, $LlamaTag) }
    } else {
        if ($have) { Step "Updating llama.cpp $have -> $LlamaTag (CUDA 13.5 build; falls back to CPU without a GPU)" }
        else { Step "Downloading llama.cpp $LlamaTag (CUDA 13.5 build, about 600 MB; falls back to CPU without a GPU)" }
        New-Item -ItemType Directory -Force $binDir | Out-Null
        $new = Join-Path $binDir 'llama.new'
        if (Test-Path $new) { Remove-Item -Recurse -Force $new }
        foreach ($name in $LlamaAssets.Keys) {
            $zip = Join-Path $binDir $name
            Get-Verified "https://github.com/ggml-org/llama.cpp/releases/download/$LlamaTag/$name" $zip $LlamaAssets[$name]
            Expand-Zip $zip $new
            Remove-Item -LiteralPath $zip -Force
        }
        if (-not (Test-Path (Join-Path $new 'llama-server.exe'))) { throw 'llama-server.exe was not found in the downloaded llama.cpp archive.' }
        [IO.File]::WriteAllText((Join-Path $new 'VERSION'), $LlamaTag)
        $old = Join-Path $binDir 'llama.old'
        if (Test-Path $old) { Remove-Item -Recurse -Force $old }
        if (Test-Path $llamaDir) {
            try { Rename-Item -LiteralPath $llamaDir -NewName 'llama.old' }
            catch { throw "Could not replace $llamaDir. Is LocalFlow still running? Quit it from the tray icon and run setup again." }
        }
        Rename-Item -LiteralPath $new -NewName 'llama'
        if (Test-Path $old) { Remove-Item -Recurse -Force $old -ErrorAction SilentlyContinue }
        Write-Host "llama.cpp $LlamaTag installed."
    }

    # ------------------------------------------------------------ models
    if ($SkipModel) {
        if (-not (Test-Path $gguf)) { Write-Host "`nSkipping the cleanup model (-SkipModel). AI cleanup needs a .gguf model in $(Join-Path $root 'models')." }
    } elseif (-not (Test-Path $gguf)) {
        Step 'Downloading the cleanup model (Qwen3.5-4B Q8_0, about 4.5 GB)'
        New-Item -ItemType Directory -Force (Join-Path $root 'models') | Out-Null
        Invoke-Direct $venvPy @('-c', 'import sys; from huggingface_hub import hf_hub_download as d; print(d(sys.argv[1], sys.argv[2], local_dir=sys.argv[3]))',
                                $GgufRepo, $GgufFile, (Join-Path $root 'models')) 'Cleanup model download'
        if (-not (Test-Path $gguf)) { throw "Expected $gguf after the download but it is not there." }
    } else { Write-Host 'Cleanup model: ok' }

    if ($script:hasGpu) { Step 'Fetching the speech model (large-v3-turbo, about 1.6 GB; skipped if already cached)' }
    else { Step 'Fetching the CPU speech model (small.en, about 0.5 GB; skipped if already cached)' }
    Invoke-Direct $venvPy @('-c', 'import sys; from huggingface_hub import snapshot_download as s; print(s(sys.argv[1]))', $whisperRepo) 'Speech model download'

    # no GPU and no config yet: start with AI cleanup off so dictation isn't held up by a slow CPU LLM
    if (-not $script:hasGpu) {
        $cfgDir = Join-Path $env:APPDATA 'LocalFlow'
        $cfgFile = Join-Path $cfgDir 'config.json'
        if (-not (Test-Path $cfgFile)) {
            New-Item -ItemType Directory -Force $cfgDir | Out-Null
            [IO.File]::WriteAllText($cfgFile, "{`n  `"cleanup`": {`"enabled`": false}`n}`n", (New-Object Text.UTF8Encoding($false)))
            Write-Host "Wrote $cfgFile with cleanup.enabled = false (turn it back on from the tray menu if you want to try it)."
        }
    }

    # ------------------------------------------------------------ autostart
    $taskOk = $false
    if ($NoTask) { Write-Host "`nSkipping the startup task (-NoTask)." }
    else {
        Step 'Registering the startup task (asks for admin once, so LocalFlow can type into elevated windows)'
        $global:LASTEXITCODE = 0
        try {
            if ($NoStart) { & (Join-Path $root 'scripts\register-task.ps1') } else { & (Join-Path $root 'scripts\register-task.ps1') -Start }
            $rc = $LASTEXITCODE
        } catch { Write-Host $_.Exception.Message; $rc = 1 }
        if ($rc -eq 0) { $taskOk = $true }
        else {
            Warn "The startup task was not registered (code $rc), so LocalFlow won't start at login."
            Write-Host '  LocalFlow still works: start it from the Start Menu. To set up the task later run:'
            Write-Host "  powershell -ExecutionPolicy Bypass -File `"$(Join-Path $root 'scripts\register-task.ps1')`""
        }
    }
    if (-not $NoStart -and -not $taskOk) {
        Start-Process -FilePath (Join-Path $root '.venv\Scripts\pythonw.exe') -ArgumentList '-m', 'localflow' -WorkingDirectory $root
        Write-Host 'Started LocalFlow.'
    }

    # ------------------------------------------------------------ tidy up
    if (Test-Path $uvCache) {
        $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
        & $uv cache clean 2>&1 | Out-Null
        $ErrorActionPreference = $prev
        Remove-Item -Recurse -Force $uvCache -ErrorAction SilentlyContinue
    }
}

# ---------------------------------------------------------------- main
try { Start-Transcript -Path $log -Force | Out-Null; $transcript = $true } catch { $transcript = $false }
$exitCode = 0
try {
    Write-Host "LocalFlow setup - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Install-LocalFlow
    Write-Host "`nLocalFlow installed. Hold Ctrl+Alt to dictate. Tray icon -> Settings." -ForegroundColor Green
    Write-Host 'First ever launch on a new GPU architecture can take ~30 s while CUDA JIT-compiles kernels (cached afterwards).'
} catch {
    $exitCode = 1
    Write-Host "`nERROR: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Setup did not finish. Nothing was rolled back; running setup again picks up where it stopped."
    Write-Host "Full log: $log"
} finally {
    if ($transcript) { try { Stop-Transcript | Out-Null } catch {} }
}
if ($exitCode -ne 0 -and $PauseOnError) { Read-Host 'Press Enter to close this window' | Out-Null }
exit $exitCode
