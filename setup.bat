@echo off
setlocal enabledelayedexpansion
:: HAL Answering Service — Windows Setup Script
:: Installs all dependencies in the correct order.
::
:: Usage:
::   setup.bat              auto-detect CUDA version
::   setup.bat cu128        force CUDA 12.8
::   setup.bat cu124        force CUDA 12.4
::   setup.bat cpu          CPU-only (no GPU)

set "CUDA_TAG=%~1"
if "%CUDA_TAG%"=="" set "CUDA_TAG=auto"
set "VENV_DIR=venv"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"

:: ── Check Python ────────────────────────────────────────
where python >nul 2>&1
if errorlevel 1 (
    echo [x] Python not found. Install Python 3.12+ from https://python.org
    exit /b 1
)

for /f "tokens=*" %%v in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2^>nul') do set "PYVER=%%v"
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do (
    set "PY_MAJOR=%%a"
    set "PY_MINOR=%%b"
)

if %PY_MAJOR% LSS 3 (
    echo [x] Python 3.12+ is required. Found: Python %PYVER%
    exit /b 1
)
if %PY_MAJOR%==3 if %PY_MINOR% LSS 12 (
    echo [x] Python 3.12+ is required. Found: Python %PYVER%
    exit /b 1
)

for /f "tokens=*" %%v in ('python --version 2^>nul') do set "PYVERSTR=%%v"
echo [+] Using %PYVERSTR%

:: ── Create venv ─────────────────────────────────────────
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [+] Creating virtual environment...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [x] Failed to create virtual environment
        exit /b 1
    )
)

:: Activate
call "%VENV_DIR%\Scripts\activate.bat"
echo [+] Virtual environment active

:: ── Detect CUDA ─────────────────────────────────────────
if "%CUDA_TAG%"=="auto" (
    where nvidia-smi >nul 2>&1
    if errorlevel 1 (
        echo [!] No NVIDIA GPU detected, installing CPU-only PyTorch
        set "CUDA_TAG=cpu"
    ) else (
        for /f "tokens=*" %%d in ('nvidia-smi --query-gpu=driver_version --format^=csv^,noheader 2^>nul') do set "DRIVER_VER=%%d"
        if "!DRIVER_VER!"=="" (
            echo [!] nvidia-smi found but no GPU detected, falling back to CPU
            set "CUDA_TAG=cpu"
        ) else (
            for /f "tokens=1 delims=." %%m in ("!DRIVER_VER!") do set "DRIVER_MAJOR=%%m"
            if !DRIVER_MAJOR! GEQ 570 (
                set "CUDA_TAG=cu128"
            ) else if !DRIVER_MAJOR! GEQ 550 (
                set "CUDA_TAG=cu124"
            ) else if !DRIVER_MAJOR! GEQ 535 (
                set "CUDA_TAG=cu121"
            ) else (
                set "CUDA_TAG=cu118"
            )
            echo [+] Detected NVIDIA driver !DRIVER_VER! -- using !CUDA_TAG!
        )
    )
)

:: ── Install PyTorch ─────────────────────────────────────
echo [+] Installing PyTorch (%CUDA_TAG%)...
if "%CUDA_TAG%"=="cpu" (
    pip install torch torchaudio --quiet
) else (
    pip install torch torchaudio --index-url "https://download.pytorch.org/whl/%CUDA_TAG%" --quiet
)
if errorlevel 1 (
    echo [x] Failed to install PyTorch
    exit /b 1
)

:: ── Install chatterbox-tts (broken deps on PyPI) ────────
:: The PyPI package pins numpy<1.26 and torch without CUDA index,
:: which breaks on Python 3.12+ and overwrites CUDA PyTorch.
:: Install the package without deps, then install its actual
:: runtime dependencies from requirements.txt.
echo [+] Installing chatterbox-tts (--no-deps)...
pip install --no-deps "chatterbox-tts>=0.1.5" --quiet
if errorlevel 1 (
    echo [x] Failed to install chatterbox-tts
    exit /b 1
)

:: ── Install everything else ─────────────────────────────
echo [+] Installing dependencies...
pip install -r requirements.txt --quiet --no-warn-conflicts
if errorlevel 1 (
    echo [x] Failed to install dependencies
    exit /b 1
)

:: ── Verify ──────────────────────────────────────────────
echo [+] Verifying installation...
python -c "import torch; print(f'  PyTorch {torch.__version__}', end=''); print(f' (CUDA {torch.version.cuda}, {torch.cuda.get_device_name(0)})' if torch.cuda.is_available() else ' (CPU only)')"
python -c "from chatterbox.tts_turbo import ChatterboxTurboTTS; print('  Chatterbox TTS OK')"
python -c "from faster_whisper import WhisperModel; print('  Faster-Whisper OK')"
python -c "from audio import mulaw_decode, SileroVAD, load_silero_model; from stt import SpeechToText; from tts import TTS; from llm import LLMClient; from config import Config; from call_handler import CallHandler; print('  All project imports OK')"

if errorlevel 1 (
    echo [x] Verification failed -- check the errors above
    exit /b 1
)

:: ── Done ────────────────────────────────────────────────
echo.
echo [+] Setup complete! Next steps:
echo   1. Start your LLM server      (e.g. LM Studio or Ollama)
echo   2. python main.py             (HAL will walk you through the rest)

