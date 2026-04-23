[CmdletBinding()]
param(
    [string]$EnvName = "cosyvoice-win-jobs"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VendorDir = Join-Path $ProjectRoot "vendor\CosyVoice"
$CondaExe = Join-Path $ProjectRoot ".conda\miniforge\Scripts\conda.exe"
if (-not (Test-Path $CondaExe)) {
    $CondaExe = "conda"
}

function Require-Command([string]$Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command not found: $Name"
    }
}

Require-Command "git"
if ($CondaExe -eq "conda") {
    Require-Command "conda"
}

if (-not (Test-Path $VendorDir)) {
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git $VendorDir
} else {
    Write-Host "CosyVoice vendor checkout already exists at $VendorDir"
}

$EnvExists = & $CondaExe env list | Select-String -Pattern "^\s*$EnvName\s"
if (-not $EnvExists) {
    & $CondaExe create -n $EnvName -y python=3.10
}

& $CondaExe run -n $EnvName python -m pip install --upgrade pip wheel "setuptools<81"
& $CondaExe run -n $EnvName python -m pip install --no-build-isolation -r (Join-Path $VendorDir "requirements.txt")
& $CondaExe run -n $EnvName python -m pip install -e $ProjectRoot

Write-Host ""
Write-Host "Bootstrap complete."
Write-Host "Next:"
Write-Host "  1. Put model files under $ProjectRoot\pretrained_models\CosyVoice2-0.5B"
Write-Host "  2. Put shared reference bundles under $ProjectRoot\shared"
Write-Host "  3. Start the server with cosyvoice-win-server.cmd"
