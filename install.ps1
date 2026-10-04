# LocalFlow installer: venv + deps + llama.cpp server + cleanup model + speech model + autostart task, then starts the app.
# Run from a normal PowerShell:  powershell -ExecutionPolicy Bypass -File install.ps1
# Flags: -NoStart (register the task but don't launch), -SkipModel (don't download the cleanup model)
param([switch]$NoStart, [switch]$SkipModel)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # the progress bar makes Invoke-WebRequest very slow in Windows PowerShell
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Step($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Warn($m) { Write-Host "WARNING: $m" -ForegroundColor Yellow }
function Check-Exit($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" } }

# ---------------------------------------------------------------- prerequisites
Step 'Checking prerequisites'

$os = [Environment]::OSVersion.Version
if ($os.Major -lt 10 -or -not [Environment]::Is64BitOperatingSystem) {
    throw 'LocalFlow needs 64-bit Windows 10 or 11.'
}
Write-Host "Windows $($os.Major).$($os.Minor).$($os.Build): ok"

$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) { $uv = Join-Path $env:USERPROFILE '.local\bin\uv.exe' }
if (-not (Test-Path $uv)) {
    throw "uv not found. Install it first (winget install --id=astral-sh.uv -e, or see https://docs.astral.sh/uv/), then open a new terminal and re-run this script."
}
Write-Host "uv: $uv"

$hasGpu = $false
$smi = (Get-Command nvidia-smi -ErrorAction SilentlyContinue).Source
if (-not $smi -and (Test-Path "$env:SystemRoot\System32\nvidia-smi.exe")) { $smi = "$env:SystemRoot\System32\nvidia-smi.exe" }
if ($smi) {
    $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    $info = & $smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>$null
    $smiExit = $LASTEXITCODE
    $ErrorActionPreference = $prev
    if ($smiExit -eq 0 -and $info) {
        $hasGpu = $true
        $first = ($info | Select-Object -First 1).Split(',') | ForEach-Object { $_.Trim() }
        Write-Host "GPU: $($first[0]), driver $($first[1]), $($first[2])"
        $drv = 0
        if ([int]::TryParse(($first[1].Split('.')[0]), [ref]$drv) -and $drv -lt 580) {
            Warn "driver $($first[1]) is older than 580. The CUDA 13 llama.cpp build needs a recent NVIDIA driver; update it from nvidia.com/drivers or the cleanup model will fall back to the CPU (or fail to load)."
        }
        $mb = 0
        if ([int]::TryParse(($first[2] -replace '[^\d]', ''), [ref]$mb) -and $mb -lt 5500) {
            Warn "only $($first[2]) of VRAM. large-v3-turbo plus the cleanup model want about 6 GB. Set stt.model to small.en in config.json if loading fails."
        }
    }
}
if (-not $hasGpu) {
    Warn 'No NVIDIA GPU / driver found (nvidia-smi is missing or failed).'
    Write-Host '  LocalFlow will still run, on the CPU: speech-to-text drops to the small.en model (int8) and'
    Write-Host '  the AI cleanup step is switched off in a fresh config (a 4B model on CPU is too slow to be useful;'
    Write-Host '  a built-in regex cleaner handles fillers and punctuation instead). Dictation will be slower than on a GPU.'
}

# ---------------------------------------------------------------- python env
Step 'Creating Python 3.12 virtual environment'
if (-not (Test-Path '.venv\Scripts\python.exe')) { & $uv venv --python 3.12 .venv; Check-Exit 'uv venv' }

Step 'Installing dependencies'
& $uv pip install --python .venv\Scripts\python.exe -r requirements.txt
Check-Exit 'uv pip install'

# ---------------------------------------------------------------- llama.cpp
$llama = Join-Path $root 'bin\llama\llama-server.exe'
if (-not (Test-Path $llama)) {
    Step 'Downloading llama.cpp (CUDA 13.4 build, works on RTX 50-series; falls back to CPU without a GPU)'
    $tag = 'b11381'
    New-Item -ItemType Directory -Force bin | Out-Null
    foreach ($f in @("llama-$tag-bin-win-cuda-13.4-x64.zip", 'cudart-llama-bin-win-cuda-13.4-x64.zip')) {
        $zip = Join-Path $root "bin\$f"
        Invoke-WebRequest "https://github.com/ggml-org/llama.cpp/releases/download/$tag/$f" -OutFile $zip
        Expand-Archive $zip -DestinationPath (Join-Path $root 'bin\llama') -Force
        Remove-Item $zip
    }
}

# ---------------------------------------------------------------- models
$gguf = Join-Path $root 'models\Qwen3.5-4B-Q4_K_M.gguf'
if (-not $SkipModel -and -not (Test-Path $gguf)) {
    Step 'Downloading the cleanup model (Qwen3.5-4B Q4_K_M, about 2.7 GB)'
    New-Item -ItemType Directory -Force models | Out-Null
    & .venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download as d; print(d('unsloth/Qwen3.5-4B-GGUF', 'Qwen3.5-4B-Q4_K_M.gguf', local_dir='models'))"
    Check-Exit 'Model download'
    if (-not (Test-Path $gguf)) { throw "Expected $gguf after the download but it is not there." }
}

if ($hasGpu) {
    Step 'Pre-fetching the speech model (large-v3-turbo, about 1.6 GB; cached if already present)'
    & .venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download as s; print(s('h2oai/faster-whisper-large-v3-turbo'))"
} else {
    Step 'Pre-fetching the CPU speech model (small.en, about 0.5 GB)'
    & .venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download as s; print(s('Systran/faster-whisper-small.en'))"
}
Check-Exit 'Speech model download'

# no GPU and no config yet: start with AI cleanup off so dictation isn't held up by a slow CPU LLM
if (-not $hasGpu) {
    $cfgDir = Join-Path $env:APPDATA 'LocalFlow'
    $cfgFile = Join-Path $cfgDir 'config.json'
    if (-not (Test-Path $cfgFile)) {
        New-Item -ItemType Directory -Force $cfgDir | Out-Null
        [System.IO.File]::WriteAllText($cfgFile, "{`n  `"cleanup`": {`"enabled`": false}`n}`n", (New-Object System.Text.UTF8Encoding($false)))
        Write-Host "Wrote $cfgFile with cleanup.enabled = false (turn it back on from the tray menu if you want to try it)."
    }
}

# ---------------------------------------------------------------- autostart
Step 'Registering autostart task (asks for admin once, so LocalFlow can type into elevated windows)'
if ($NoStart) { & "$root\scripts\register-task.ps1" } else { & "$root\scripts\register-task.ps1" -Start }

Write-Host "`nLocalFlow installed. Hold Ctrl+Alt to dictate. Tray icon -> Settings." -ForegroundColor Green
Write-Host 'First ever launch on a new GPU architecture can take ~30 s while CUDA JIT-compiles kernels (cached afterwards).'
