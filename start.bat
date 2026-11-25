@echo off
rem Start the classic text UI or the native multimodal HTTP service.
rem Usage: start.bat [--multimodal] [port]
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "VENV=.venv"
set "EXTRAS=serve"
set "TORCH_PACKAGES=torch"
if /i "%~1"=="--multimodal" (
    set "VENV=.venv-mm"
    set "EXTRAS=multimodal,serve"
    set "TORCH_PACKAGES=torch torchvision"
    shift
)
set "PORT=%~1"
if "%PORT%"=="" set "PORT=8000"

if "%EXTRAS%"=="multimodal,serve" if not defined LAYA_MULTIMODAL_MODEL set "LAYA_MULTIMODAL_MODEL=%CD%\models\jev-multimodal-qwen3.5-4b"
if "%EXTRAS%"=="multimodal,serve" if not exist "%LAYA_MULTIMODAL_MODEL%\multimodal_config.json" (
    echo [error] Set LAYA_MULTIMODAL_MODEL to a trained local checkpoint directory.
    exit /b 2
)

if not defined TORCH_INDEX_URL set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu"
if not exist "%VENV%\Scripts\python.exe" (
    rem Any interpreter the package supports, newest first. Pinning one version fails on a
    rem machine that has a different supported one: pyproject declares >=3.10 and LOCAL_SETUP.md
    rem documents a 3.11 venv, so a hard py -3.12 could not set up the repo it ships in.
    echo [setup] Creating virtual environment...
    set "PYEXE="
    for %%V in (3.13 3.12 3.11 3.10) do if not defined PYEXE (py -%%V -c "pass" >nul 2>&1 && set "PYEXE=py -%%V")
    if not defined PYEXE (
        echo [error] No supported Python found. Install Python 3.10 or newer from python.org.
        exit /b 1
    )
    !PYEXE! -m venv "%VENV%" || goto :fail
    "%VENV%\Scripts\python.exe" -m pip install -q --upgrade pip || goto :fail
    "%VENV%\Scripts\python.exe" -m pip install -q %TORCH_PACKAGES% --index-url "%TORCH_INDEX_URL%" || goto :fail
    "%VENV%\Scripts\python.exe" -m pip install -q -e ".[%EXTRAS%]" || goto :fail
)

set PYTHONUTF8=1
if "%EXTRAS%"=="multimodal,serve" goto :native
echo Laya UI: http://localhost:%PORT%   (Ctrl+C to stop)
start "" /b cmd /c "timeout /t 8 /nobreak >nul & start http://localhost:%PORT%"
"%VENV%\Scripts\python.exe" examples\server.py --host 127.0.0.1 --port %PORT%
exit /b %ERRORLEVEL%

:native
set "LAYA_HOST=127.0.0.1"
set "LAYA_PORT=%PORT%"
if not defined LAYA_PRELOAD set "LAYA_PRELOAD=0"
echo Laya media API: http://localhost:%PORT%/v1/multimodal
"%VENV%\Scripts\python.exe" -m laya.serve
exit /b %ERRORLEVEL%

:fail
echo [error] Setup failed. See the messages above.
exit /b 1
