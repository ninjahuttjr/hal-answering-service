"""Entry point — load environment, initialize models, start the server."""

# ── Suppress third-party warning noise before any imports ──
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")

import os
os.environ["NUMEXPR_MAX_THREADS"] = os.environ.get("NUMEXPR_MAX_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")  # suppress "NumExpr defaulting to N threads" msg

import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)-20s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hal")

# Quiet down chatty third-party loggers
for _logger_name in ("faster_whisper", "httpx", "httpcore", "uvicorn", "uvicorn.access",
                      "uvicorn.error", "numexpr", "stt", "llm"):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)


def _preflight():
    """Check for common setup issues and print helpful errors instead of tracebacks."""
    errors = []

    # Check .env file exists
    if not os.path.exists(".env"):
        errors.append(
            "No .env file found.\n"
            "  Copy the example config and fill in your values:\n"
            "    Windows:     copy .env.example .env\n"
            "    Linux/macOS: cp .env.example .env"
        )

    # Check Python version
    if sys.version_info < (3, 12):
        errors.append(f"Python 3.12+ required, but you have {sys.version_info.major}.{sys.version_info.minor}.")

    # Check PyTorch + CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            errors.append(
                "PyTorch cannot find CUDA.\n"
                "  Make sure you installed PyTorch with the correct CUDA index:\n"
                "    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124\n"
                "  If you don't have an NVIDIA GPU, set STT_DEVICE=cpu in .env."
            )
    except ImportError:
        errors.append("PyTorch is not installed. Run the setup script: setup.bat (Windows) or ./setup.sh (Linux/macOS)")

    # Check chatterbox version
    try:
        from chatterbox.tts_turbo import ChatterboxTurboTTS  # noqa: F401
    except ImportError:
        try:
            import chatterbox
            ver = getattr(chatterbox, "__version__", "unknown")
            errors.append(
                f"chatterbox-tts {ver} is too old (missing tts_turbo module).\n"
                "  Upgrade to >=0.1.5:\n"
                '    pip install --no-deps "chatterbox-tts>=0.1.5"'
            )
        except ImportError:
            errors.append(
                "chatterbox-tts is not installed.\n"
                "  Install it with:\n"
                '    pip install --no-deps "chatterbox-tts>=0.1.5"'
            )

    # Check required env vars (only if .env exists)
    if os.path.exists(".env"):
        from dotenv import load_dotenv
        load_dotenv()
        required = {
            "SIGNALWIRE_PROJECT_ID": "SignalWire project ID (from dashboard)",
            "SIGNALWIRE_TOKEN": "SignalWire API token",
            "SIGNALWIRE_SPACE": "SignalWire space name",
            "SIGNALWIRE_PHONE_NUMBER": "SignalWire phone number",
            "PUBLIC_HOST": "Public HTTPS hostname",
        }
        missing = []
        for key, desc in required.items():
            val = os.environ.get(key, "")
            if not val or val.startswith("your-") or val == "+1XXXXXXXXXX":
                missing.append(f"    {key} -- {desc}")
        if missing:
            errors.append("Missing required settings in .env:\n" + "\n".join(missing))

    if errors:
        print("\n" + "=" * 60)
        print("  SETUP ISSUES DETECTED")
        print("=" * 60)
        for i, err in enumerate(errors, 1):
            print(f"\n  {i}. {err}")
        print("\n" + "=" * 60)
        print("  Fix the above and try again.")
        print("  See README.md for full setup instructions.")
        print("=" * 60 + "\n")
        sys.exit(1)


BANNER = r"""
  ██╗  ██╗ █████╗ ██╗
  ██║  ██║██╔══██╗██║
  ███████║███████║██║
  ██╔══██║██╔══██║██║
  ██║  ██║██║  ██║███████╗
  ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝
  Answering Service
"""


def main():
    _preflight()

    import uvicorn
    from dotenv import load_dotenv
    from config import Config
    from audio import load_silero_model
    from stt import SpeechToText
    from tts import TTS
    from server import create_app

    print(BANNER)
    t_start = time.perf_counter()
    log.info("Starting up...")

    load_dotenv()
    config = Config()
    log.info("Config loaded (LLM: %s, STT: %s)", config.llm_model, config.stt_model)

    # Load STT
    t0 = time.perf_counter()
    stt = SpeechToText(
        model_size=config.stt_model,
        device=config.stt_device,
        compute_type=config.stt_compute_type,
    )
    stt.load()
    log.info("STT loaded in %.1fs", time.perf_counter() - t0)

    # Load TTS (kept hot on GPU)
    t0 = time.perf_counter()
    voice_prompt = config.tts_voice_prompt or None
    tts = TTS(voice_prompt=voice_prompt, device="cuda")
    log.info("TTS loaded in %.1fs", time.perf_counter() - t0)

    # Load VAD (shared across calls, deep-copied per call)
    t0 = time.perf_counter()
    vad_model = load_silero_model()
    log.info("VAD loaded in %.1fs", time.perf_counter() - t0)

    # Pre-record greetings so first pickup is instant
    t0 = time.perf_counter()
    from prompts import get_greeting, SILENCE_PROMPTS
    greeting_cache = {}
    owner = config.owner_name
    for period in ("morning", "afternoon", "evening"):
        text = get_greeting(owner, time_of_day=period)
        mulaw = b"".join(c["mulaw"] for c in tts.synthesize_mulaw_streaming(text))
        greeting_cache[period] = {"text": text, "mulaw": mulaw}
        log.info("Pre-recorded greeting (%s): %.1fs", period, len(mulaw) / 8000)
    log.info("Greetings pre-recorded in %.1fs", time.perf_counter() - t0)

    # Pre-record silence prompts
    t0 = time.perf_counter()
    silence_prompt_cache = []
    for prompt_text in SILENCE_PROMPTS:
        mulaw = b"".join(c["mulaw"] for c in tts.synthesize_mulaw_streaming(prompt_text))
        silence_prompt_cache.append({"text": prompt_text, "mulaw": mulaw})
    log.info("Silence prompts pre-recorded in %.1fs", time.perf_counter() - t0)

    os.makedirs(config.recordings_dir, exist_ok=True)

    app = create_app(config, stt, tts, vad_model, greeting_cache, silence_prompt_cache)

    total = time.perf_counter() - t_start
    log.info("Ready in %.1fs — listening on %s:%d", total, config.host, config.port)

    uvicorn.run(app, host=config.host, port=config.port, log_level="warning",
                ws_max_size=65536)


if __name__ == "__main__":
    main()
