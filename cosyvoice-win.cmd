@echo off
setlocal EnableExtensions EnableDelayedExpansion
set "PROJECT_ROOT=%~dp0"
set "ENV_NAME=cosyvoice-win-jobs"
if not "%COSYVOICE_ENV_NAME%"=="" set "ENV_NAME=%COSYVOICE_ENV_NAME%"
set "CONDA_EXE=%PROJECT_ROOT%.conda\miniforge\Scripts\conda.exe"
if not exist "%CONDA_EXE%" set "CONDA_EXE=conda"

set "PYTHONPATH=%PROJECT_ROOT%src;%PROJECT_ROOT%vendor\CosyVoice;%PROJECT_ROOT%vendor\CosyVoice\third_party\Matcha-TTS"
if "%PYTHONUTF8%"=="" set "PYTHONUTF8=1"
if "%PYTHONIOENCODING%"=="" set "PYTHONIOENCODING=utf-8"
set "TEMP=%PROJECT_ROOT%.tmp"
set "TMP=%PROJECT_ROOT%.tmp"
set "XDG_DATA_HOME=%PROJECT_ROOT%.data"

if not exist "%TEMP%" mkdir "%TEMP%"
if not exist "%XDG_DATA_HOME%" mkdir "%XDG_DATA_HOME%"

"%CONDA_EXE%" run -n "%ENV_NAME%" python -m cosyvoice_win.cli %*
exit /b %ERRORLEVEL%
