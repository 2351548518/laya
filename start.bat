@echo off
rem Start the native multimodal HTTP service.
rem Usage: start.bat [port]
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "VENV=.venv-mm"
set "PORT=%~1"
if "%PORT%"=="" set "PORT=8000"

if not defined LAYA_MULTIMODAL_MODEL set "LAYA_MULTIMODAL_MODEL=%CD%\models\jev-multimodal-qwen3.5-4b"
if not exist "%LAYA_MULTIMODAL_MODEL%\multimodal_config.json" (
    echo [error] Set LAYA_MULTIMODAL_MODEL to a trained local checkpoint directory.
    exit /b 2
)

if not defined TORCH_INDEX_URL set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu"
if not exist "%VENV%\Scripts\python.exe" (
    rem Select an installed Python version for the native runtime.
    echo [setup] Creating virtual environment...
    set "PYEXE="
    for %%V in (3.13 3.12 3.11) do if not defined PYEXE (py -%%V -c "pass" >nul 2>&1 && set "PYEXE=py -%%V")
    if not defined PYEXE (
        echo [error] No supported Python found. Install Python 3.11 or newer from python.org.
        exit /b 1
    )
    !PYEXE! -m venv "%VENV%" || goto :fail
)
if not exist "%VENV%\Scripts\laya-multimodal-serve.exe" (
    echo [setup] Installing Laya dependencies...
    "%VENV%\Scripts\python.exe" -m pip install -q --upgrade pip || goto :fail
    "%VENV%\Scripts\python.exe" -m pip install -q torch torchvision --index-url "%TORCH_INDEX_URL%" || goto :fail
    "%VENV%\Scripts\python.exe" -m pip install -q -e ".[multimodal,serve]" || goto :fail
)

set PYTHONUTF8=1
set "LAYA_HOST=127.0.0.1"
set "LAYA_PORT=%PORT%"
if not defined LAYA_PRELOAD set "LAYA_PRELOAD=0"
echo Laya media API: http://localhost:%PORT%/v1/multimodal
"%VENV%\Scripts\python.exe" -m laya_multimodal.serve
exit /b %ERRORLEVEL%

:fail
echo [error] Setup failed. See the messages above.
exit /b 1
