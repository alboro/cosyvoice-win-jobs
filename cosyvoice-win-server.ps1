$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvName = if ($env:COSYVOICE_ENV_NAME) { $env:COSYVOICE_ENV_NAME } else { "cosyvoice-win-jobs" }
$CondaExe = Join-Path $ProjectRoot ".conda\miniforge\Scripts\conda.exe"
if (-not (Test-Path $CondaExe)) {
    $CondaExe = "conda"
}

$env:PYTHONPATH = "$ProjectRoot\src;$ProjectRoot\vendor\CosyVoice;$ProjectRoot\vendor\CosyVoice\third_party\Matcha-TTS"
if (-not $env:PYTHONUTF8) { $env:PYTHONUTF8 = "1" }
if (-not $env:PYTHONIOENCODING) { $env:PYTHONIOENCODING = "utf-8" }
$env:TEMP = "$ProjectRoot\.tmp"
$env:TMP = "$ProjectRoot\.tmp"
$env:XDG_DATA_HOME = "$ProjectRoot\.data"

New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
New-Item -ItemType Directory -Force -Path $env:XDG_DATA_HOME | Out-Null

& $CondaExe run -n $EnvName python -m cosyvoice_win.server @args
