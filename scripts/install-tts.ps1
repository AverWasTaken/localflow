# LocalFlow: installs the voice for spoken command-mode answers (Qwen3-TTS 0.6B) into its own environment.
#   .venv-tts\                                    torch 2.8 (CUDA 12.8) + faster-qwen3-tts, about 7.7 GB
#   models\tts\Qwen3-TTS-12Hz-0.6B-CustomVoice\   the model, about 2.5 GB
# NVIDIA GPU only. install.ps1 runs this (skip it there with -SkipTts); you can also run it by hand:
#   powershell -ExecutionPolicy Bypass -File scripts\install-tts.ps1
# Safe to run again: it skips what is already in place (-Force reinstalls the packages).
# Exit code 0 = installed, 3 = no NVIDIA GPU, 1 = failed (LocalFlow keeps working without spoken answers).
param([string]$Uv = '', [switch]$Force)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$root = Split-Path -Parent $PSScriptRoot

# ---------------------------------------------------------------- pinned versions
$TorchVersion = '2.8.0+cu128'          # has sm_120 kernels (RTX 50xx)
$TorchIndex   = 'https://download.pytorch.org/whl/cu128'
$TtsRepo      = 'Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice'
$TtsRevision  = '85e237c12c027371202489a0ec509ded67b5e4b5'
$TtsDirName   = 'Qwen3-TTS-12Hz-0.6B-CustomVoice'

$venv     = Join-Path $root '.venv-tts'
$venvPy   = Join-Path $venv 'Scripts\python.exe'
$reqs     = Join-Path $root 'requirements-tts.txt'
$modelDir = Join-Path $root "models\tts\$TtsDirName"
$marker   = Join-Path $venv 'localflow-tts.txt'

function Say($m) { Write-Host "[tts] $m" }

function Invoke-Checked([string]$Exe, [string[]]$Arguments, [string]$What) {
    $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try {
        & $Exe @Arguments 2>&1 | ForEach-Object { Write-Host "$_" }
        $code = $LASTEXITCODE
    } finally { $ErrorActionPreference = $prev }
    if ($code -ne 0) { throw "$What failed (exit code $code)." }
}

function Install-Tts {
    $smi = (Get-Command nvidia-smi -ErrorAction SilentlyContinue).Source
    if (-not $smi -and (Test-Path "$env:SystemRoot\System32\nvidia-smi.exe")) { $smi = "$env:SystemRoot\System32\nvidia-smi.exe" }
    if (-not $smi) { Say 'No NVIDIA GPU found: spoken answers need one. Nothing installed.'; return 3 }
    if (-not (Test-Path $reqs)) { throw "$reqs is missing." }

    # uv: the caller's, then LocalFlow's private copy (install.ps1), then one on PATH
    if (-not $Uv) {
        $priv = Join-Path $root 'bin\uv\uv.exe'
        if (Test-Path $priv) { $Uv = $priv }
        else { $Uv = (Get-Command uv -ErrorAction SilentlyContinue).Source }
    }
    if (-not $Uv -or -not (Test-Path $Uv)) { throw 'uv was not found (run install.ps1 first).' }
    if (Test-Path (Join-Path $root 'bin\uv\uv.exe')) {
        # LocalFlow's own Python lives in python\ (same settings as install.ps1)
        if (-not $env:UV_PYTHON_INSTALL_DIR) { $env:UV_PYTHON_INSTALL_DIR = Join-Path $root 'python' }
        if (-not $env:UV_PYTHON_PREFERENCE) { $env:UV_PYTHON_PREFERENCE = 'only-managed' }
    }
    $env:UV_NO_PROGRESS = '1'
    # same interpreter as the app's .venv when there is one (a dev checkout may use a system Python 3.12)
    $pySpec = '3.12'
    $cfg = Join-Path $root '.venv\pyvenv.cfg'
    if (-not $env:UV_PYTHON_INSTALL_DIR -and (Test-Path $cfg)) {
        $homeLine = Get-Content -LiteralPath $cfg | Where-Object { $_ -match '^\s*home\s*=' } | Select-Object -First 1
        if ($homeLine) {
            $exe = Join-Path (($homeLine -split '=', 2)[1].Trim()) 'python.exe'
            if (Test-Path $exe) { $pySpec = $exe }
        }
    }

    $stamp = "torch=$TorchVersion;reqs=$((Get-FileHash -LiteralPath $reqs -Algorithm SHA256).Hash);model=$TtsRevision"
    $pkgsOk = (-not $Force) -and (Test-Path $venvPy) -and (Test-Path $marker) -and
              ((Get-Content -LiteralPath $marker -Raw).Trim() -eq $stamp)
    if ($pkgsOk) {
        $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
        & $venvPy -c 'import torch, faster_qwen3_tts, sounddevice' 2>$null
        $pkgsOk = ($LASTEXITCODE -eq 0)
        $ErrorActionPreference = $prev
    }

    if ($pkgsOk) { Say 'Python packages: ok' }
    else {
        if (Test-Path $venvPy) {
            $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
            & $venvPy -c 'import sys' 2>$null
            $ok = ($LASTEXITCODE -eq 0)
            $ErrorActionPreference = $prev
            if (-not $ok) { Say 'The existing .venv-tts is broken; recreating it.'; Remove-Item -Recurse -Force $venv }
        }
        if (-not (Test-Path $venvPy)) {
            Say "Creating .venv-tts (Python $pySpec)"
            Invoke-Checked $Uv @('venv', '--python', $pySpec, $venv) 'Creating .venv-tts'
        }
        Remove-Item -LiteralPath $marker -Force -ErrorAction SilentlyContinue
        Say "Installing torch $TorchVersion (CUDA 12.8, about 3 GB download)"
        $tv = $TorchVersion
        Invoke-Checked $Uv @('pip', 'install', '--python', $venvPy, "torch==$tv", "torchaudio==$tv",
                             '--index-url', $TorchIndex) 'Installing torch'
        Say 'Installing faster-qwen3-tts and its packages'
        $cons = Join-Path $venv 'torch-constraints.txt'
        [IO.File]::WriteAllText($cons, "torch==$tv`ntorchaudio==$tv`n")
        Invoke-Checked $Uv @('pip', 'install', '--python', $venvPy, '-r', $reqs, '-c', $cons) 'Installing the TTS packages'
    }

    # the CUDA build must carry kernels for this GPU (RTX 50xx = sm_120)
    # (no double quotes in here: Windows PowerShell 5.1 strips them from native-command arguments)
    $check = @'
import sys, torch
arch = torch.cuda.get_arch_list()
assert 'sm_120' in arch, f'torch {torch.__version__} has no sm_120 kernels: {arch}'
if not torch.cuda.is_available():
    sys.exit('CUDA is not available to torch (update the NVIDIA driver)')
cap = 'sm_%d%d' % torch.cuda.get_device_capability(0)
if cap not in arch:
    print(f'WARNING: GPU {cap} not in the torch build {arch}; it may not run')
import faster_qwen3_tts, sounddevice
print('torch', torch.__version__, 'on', torch.cuda.get_device_name(0), cap)
'@
    Invoke-Checked $venvPy @('-c', $check) 'Checking torch / CUDA'

    if ((Test-Path (Join-Path $modelDir 'config.json')) -and (Test-Path (Join-Path $modelDir 'model.safetensors')) -and
        (Test-Path (Join-Path $modelDir 'speech_tokenizer\model.safetensors')) -and $pkgsOk) {
        Say 'Voice model: ok'
    } else {
        Say "Downloading the voice model ($TtsRepo, about 2.5 GB)"
        New-Item -ItemType Directory -Force (Split-Path -Parent $modelDir) | Out-Null
        $dl = 'import sys; from huggingface_hub import snapshot_download as s; print(s(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3]))'
        $prevOff = $env:HF_HUB_OFFLINE; $env:HF_HUB_OFFLINE = '0'
        try { Invoke-Checked $venvPy @('-c', $dl, $TtsRepo, $TtsRevision, $modelDir) 'Voice model download' }
        finally { $env:HF_HUB_OFFLINE = $prevOff }
        foreach ($f in 'config.json', 'model.safetensors', 'speech_tokenizer\model.safetensors') {
            if (-not (Test-Path (Join-Path $modelDir $f))) { throw "Expected $f in $modelDir after the download." }
        }
    }
    [IO.File]::WriteAllText($marker, $stamp)
    Say 'Spoken answers installed.'
    return 0
}

try { $rc = Install-Tts }
catch {
    Write-Host "[tts] ERROR: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host '[tts] LocalFlow works without spoken answers. Run scripts\install-tts.ps1 again to retry.'
    $rc = 1
}
exit $rc
