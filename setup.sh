#!/usr/bin/env bash
# HAL Answering Service — Setup Script
# Installs all dependencies in the correct order.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh              # auto-detect CUDA version
#   ./setup.sh cu124        # force CUDA 12.4
#   ./setup.sh cpu          # CPU-only (no GPU)

set -euo pipefail
export PIP_DISABLE_PIP_VERSION_CHECK=1

CUDA_TAG="${1:-auto}"
VENV_DIR="venv"

# ── Colors ──────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[+]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
error() { echo -e "${RED}[x]${NC} $*"; exit 1; }

# ── Safety checks ─────────────────────────────────────────
if [ "${EUID:-$(id -u)}" -eq 0 ]; then
    error "Do not run setup with sudo/root. Run as your normal user."
fi

# ── Check Python ────────────────────────────────────────
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 12 ]; then
            PYTHON="$cmd"
            break
        fi
    fi
done
[ -z "$PYTHON" ] && error "Python 3.12+ is required. Found: $(python3 --version 2>&1 || echo 'none')"
info "Using $PYTHON ($($PYTHON --version))"

# ── Create venv ─────────────────────────────────────────
if [ -d "$VENV_DIR" ] && [ ! -w "$VENV_DIR" ]; then
    error "Existing $VENV_DIR is not writable (likely created with sudo). Remove it with: sudo rm -rf $VENV_DIR"
fi

if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment..."
    $PYTHON -m venv "$VENV_DIR"
fi

# Activate
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
elif [ -f "$VENV_DIR/Scripts/activate" ]; then
    source "$VENV_DIR/Scripts/activate"
else
    error "Could not find venv activate script"
fi
info "Virtual environment active"

# ── Detect CUDA ─────────────────────────────────────────
if [ "$CUDA_TAG" = "auto" ]; then
    if command -v nvidia-smi &>/dev/null; then
        CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "")
        if [ -n "$CUDA_VER" ]; then
            # Map driver version to CUDA toolkit version
            # Driver 570+ → CUDA 12.8, 550+ → CUDA 12.4, 535+ → CUDA 12.1
            DRIVER_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
            if [ "$DRIVER_MAJOR" -ge 570 ]; then
                CUDA_TAG="cu128"
            elif [ "$DRIVER_MAJOR" -ge 550 ]; then
                CUDA_TAG="cu124"
            elif [ "$DRIVER_MAJOR" -ge 535 ]; then
                CUDA_TAG="cu121"
            else
                CUDA_TAG="cu118"
            fi
            info "Detected NVIDIA driver $CUDA_VER → using $CUDA_TAG"
        else
            warn "nvidia-smi found but no GPU detected, falling back to CPU"
            CUDA_TAG="cpu"
        fi
    else
        warn "No NVIDIA GPU detected, installing CPU-only PyTorch"
        CUDA_TAG="cpu"
    fi
fi

# ── Install PyTorch ─────────────────────────────────────
info "Installing PyTorch ($CUDA_TAG)..."
if [ "$CUDA_TAG" = "cpu" ]; then
    pip install torch torchaudio --quiet
else
    pip install torch torchaudio --index-url "https://download.pytorch.org/whl/$CUDA_TAG" --quiet
fi

# ── Install chatterbox-tts (broken deps on PyPI) ────────
# The PyPI package pins numpy<1.26 and torch without CUDA index,
# which breaks on Python 3.12+ and overwrites CUDA PyTorch.
# Install the package without deps, then install its actual
# runtime dependencies from requirements.txt.
info "Installing chatterbox-tts (--no-deps)..."
pip install --no-deps 'chatterbox-tts>=0.1.5' --quiet

# ── Install everything else ─────────────────────────────
info "Installing dependencies..."
pip install -r requirements.txt --quiet --no-warn-conflicts

# If .env exists and we're on CPU, normalize runtime defaults.
if [ "$CUDA_TAG" = "cpu" ] && [ -f ".env" ]; then
    info "Applying CPU-friendly defaults in .env (STT/TTS auto mode)..."
    $PYTHON - <<'PY'
from pathlib import Path

p = Path(".env")
lines = p.read_text(encoding="utf-8").splitlines()

seen = {"STT_DEVICE": False, "STT_COMPUTE_TYPE": False, "TTS_DEVICE": False}
out = []
for line in lines:
    if line.startswith("STT_DEVICE="):
        out.append("STT_DEVICE=auto")
        seen["STT_DEVICE"] = True
    elif line.startswith("STT_COMPUTE_TYPE="):
        out.append("STT_COMPUTE_TYPE=auto")
        seen["STT_COMPUTE_TYPE"] = True
    elif line.startswith("TTS_DEVICE="):
        out.append("TTS_DEVICE=auto")
        seen["TTS_DEVICE"] = True
    else:
        out.append(line)

if not seen["STT_DEVICE"]:
    out.append("STT_DEVICE=auto")
if not seen["STT_COMPUTE_TYPE"]:
    out.append("STT_COMPUTE_TYPE=auto")
if not seen["TTS_DEVICE"]:
    out.append("TTS_DEVICE=auto")

p.write_text("\n".join(out) + "\n", encoding="utf-8")
PY
fi

# ── Verify ──────────────────────────────────────────────
info "Verifying installation..."
$PYTHON -c "
import torch
print(f'  PyTorch {torch.__version__}', end='')
if torch.cuda.is_available():
    print(f' (CUDA {torch.version.cuda}, {torch.cuda.get_device_name(0)})')
else:
    print(' (CPU only)')

from chatterbox.tts_turbo import ChatterboxTurboTTS
print('  Chatterbox TTS OK')

from faster_whisper import WhisperModel
print('  Faster-Whisper OK')

from audio import mulaw_decode, SileroVAD, load_silero_model
from stt import SpeechToText
from tts import TTS
from llm import LLMClient
from config import Config
from call_handler import CallHandler
print('  All project imports OK')
"

# ── Done ────────────────────────────────────────────────
echo ""
info "Setup complete! Next steps:"
echo "  1. Start your LLM server   # e.g. LM Studio or Ollama"
echo "  2. python main.py          # HAL will walk you through the rest"
