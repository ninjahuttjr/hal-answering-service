"""Entry point — load environment, initialize models, start the server."""

# ── Suppress third-party warning noise before any imports ──
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")

import os
os.environ["NUMEXPR_MAX_THREADS"] = os.environ.get("NUMEXPR_MAX_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")  # suppress "NumExpr defaulting to N threads" msg

import argparse
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


VALID_DEVICE_VALUES = {"auto", "cuda", "cpu"}


def _normalize_choice(var_name: str, value: str, valid_values: set[str]) -> str:
    normalized = (value or "").strip().lower()
    if not normalized:
        normalized = "auto"
    if normalized not in valid_values:
        allowed = ", ".join(sorted(valid_values))
        raise ValueError(f"{var_name} must be one of: {allowed}. Got: {value!r}")
    return normalized


def _resolve_runtime(stt_device: str, stt_compute_type: str, tts_device: str, cuda_available: bool):
    """Resolve effective runtime settings from user preferences + hardware."""
    stt_pref = _normalize_choice("STT_DEVICE", stt_device, VALID_DEVICE_VALUES)
    tts_pref = _normalize_choice("TTS_DEVICE", tts_device, VALID_DEVICE_VALUES)
    compute_pref = (stt_compute_type or "auto").strip().lower() or "auto"

    resolved_stt_device = "cuda" if stt_pref == "auto" and cuda_available else stt_pref
    resolved_tts_device = "cuda" if tts_pref == "auto" and cuda_available else tts_pref
    if stt_pref == "auto" and not cuda_available:
        resolved_stt_device = "cpu"
    if tts_pref == "auto" and not cuda_available:
        resolved_tts_device = "cpu"

    warnings = []
    if resolved_stt_device == "cuda" and not cuda_available:
        warnings.append("STT_DEVICE requested CUDA, but CUDA is unavailable. Falling back to CPU.")
        resolved_stt_device = "cpu"
    if resolved_tts_device == "cuda" and not cuda_available:
        warnings.append("TTS_DEVICE requested CUDA, but CUDA is unavailable. Falling back to CPU.")
        resolved_tts_device = "cpu"

    if resolved_stt_device == "cpu":
        if compute_pref in ("auto", "float16"):
            resolved_compute_type = "int8"
            warnings.append(
                "Using STT on CPU; forcing STT_COMPUTE_TYPE=int8 "
                "(float16 is CUDA-oriented and may fail or be slow on CPU)."
            )
        else:
            resolved_compute_type = compute_pref
    else:
        resolved_compute_type = "float16" if compute_pref == "auto" else compute_pref

    return resolved_stt_device, resolved_compute_type, resolved_tts_device, warnings


def _preflight():
    """Check for common setup issues and print helpful errors instead of tracebacks."""
    errors = []
    from config import Config

    load_dotenv = None
    try:
        from dotenv import load_dotenv
    except ImportError:
        errors.append(
            "python-dotenv is not installed. Run the setup script: "
            "setup.bat (Windows) or ./setup.sh (Linux/macOS)"
        )

    # Check .env file exists
    has_dotenv = os.path.exists(".env")
    if not has_dotenv:
        errors.append(
            "No .env file found.\n"
            "  Copy the example config and fill in your values:\n"
            "    Windows:     copy .env.example .env\n"
            "    Linux/macOS: cp .env.example .env"
        )
    elif load_dotenv:
        load_dotenv()

    # Check Python version
    if sys.version_info < (3, 12):
        errors.append(f"Python 3.12+ required, but you have {sys.version_info.major}.{sys.version_info.minor}.")

    # Check PyTorch + CUDA
    torch = None
    try:
        import torch
    except ImportError:
        errors.append("PyTorch is not installed. Run the setup script: setup.bat (Windows) or ./setup.sh (Linux/macOS)")

    config = Config()
    if torch is not None:
        try:
            _resolve_runtime(
                config.stt_device, config.stt_compute_type, config.tts_device, torch.cuda.is_available()
            )
        except ValueError as e:
            errors.append(str(e))

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
    if has_dotenv:
        required = {
            "SIGNALWIRE_PROJECT_ID": "SignalWire project ID (from dashboard)",
            "SIGNALWIRE_TOKEN": "SignalWire API token",
            "SIGNALWIRE_SPACE": "SignalWire space name",
            "SIGNALWIRE_PHONE_NUMBER": "SignalWire phone number",
            "PUBLIC_HOST": "Public HTTPS hostname",
            "OWNER_NAME": "Name HAL uses in greetings",
        }
        missing = []
        for key, desc in required.items():
            val = os.environ.get(key, "")
            is_placeholder = (
                val.startswith("your-")
                or val == "+1XXXXXXXXXX"
                or val == "YourName"
            )
            if not val or is_placeholder:
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

BANNER_DEMO = r"""
  ██╗  ██╗ █████╗ ██╗
  ██║  ██║██╔══██╗██║
  ███████║███████║██║
  ██╔══██║██╔══██║██║
  ██║  ██║██║  ██║███████╗
  ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝
  Answering Service — DEMO MODE
"""


def _infer_llm_provider_from_url(base_url: str) -> str:
    url = (base_url or "").strip().lower()
    if "127.0.0.1:1234" in url or "localhost:1234" in url:
        return "lmstudio"
    if "127.0.0.1:11434" in url or "localhost:11434" in url or "ollama" in url:
        return "ollama"
    return "openai_compatible"


def _demo_interactive_setup():
    """Prompt for LLM settings if no .env exists or LLM fields are defaults.
    Writes a minimal .env so settings persist for next run.
    """
    env_path = os.path.join(os.path.dirname(__file__) or ".", ".env")
    has_dotenv = os.path.exists(env_path)

    # Load existing .env if present
    if has_dotenv:
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        except ImportError:
            pass

    # Check if demo settings look configured
    llm_provider = os.environ.get("LLM_PROVIDER", "").strip()
    llm_url = os.environ.get("LLM_BASE_URL", "").strip()
    llm_key = os.environ.get("LLM_API_KEY", "").strip()
    llm_model = os.environ.get("LLM_MODEL", "").strip()
    owner = os.environ.get("OWNER_NAME", "").strip()

    needs_setup = not has_dotenv or not llm_url or not owner
    if not needs_setup:
        return  # Already configured

    print()
    print("  First-time setup — just a few questions.\n")

    def _ask(prompt: str, default: str) -> str:
        display = f"  {prompt} [{default}]: " if default else f"  {prompt}: "
        answer = input(display).strip()
        return answer if answer else default

    if not llm_provider:
        llm_provider = _infer_llm_provider_from_url(llm_url) if llm_url else "lmstudio"

    print("  LLM provider:")
    print("    1) LM Studio")
    print("    2) Ollama")
    print("    3) Other OpenAI-compatible server")
    default_choice = "1"
    if llm_provider == "ollama":
        default_choice = "2"
    elif llm_provider == "openai_compatible":
        default_choice = "3"
    provider_choice = _ask("Choose provider", default_choice)
    if provider_choice == "2":
        llm_provider = "ollama"
    elif provider_choice == "3":
        llm_provider = "openai_compatible"
    else:
        llm_provider = "lmstudio"

    default_url = llm_url
    default_key = llm_key
    default_model = llm_model
    model_prompt = "LLM model name"
    if llm_provider == "lmstudio":
        default_url = default_url or "http://127.0.0.1:1234/v1"
        default_key = default_key or "lm-studio"
        model_prompt = "LLM model name (leave blank for server default)"
    elif llm_provider == "ollama":
        default_url = default_url or "http://127.0.0.1:11434/v1"
        default_key = default_key or "ollama"
        default_model = default_model or "qwen3:4b"
        model_prompt = "Ollama model name (must match `ollama list`, e.g. qwen3:4b)"
    else:
        default_url = default_url or "http://127.0.0.1:1234/v1"
        default_key = default_key or "local"
        model_prompt = "LLM model name"

    llm_url = _ask("LLM server URL", default_url)
    llm_key = _ask("LLM API key", default_key)
    llm_model = _ask(model_prompt, default_model)
    owner_name = _ask("Your name (for greetings)", owner or "Dave")

    # Auto-detect bundled TTS model dir (no prompt needed)
    tts_model_dir = os.environ.get("TTS_MODEL_DIR", "").strip()
    bundled_default = os.path.join(os.path.dirname(__file__) or ".", "models", "chatterbox")
    if not tts_model_dir and os.path.isdir(bundled_default):
        tts_model_dir = bundled_default

    # Auto-detect HAL voice prompt (ship hal9000.wav as default)
    tts_voice = os.environ.get("TTS_VOICE_PROMPT", "").strip()
    if not tts_voice:
        hal_wav = os.path.join(os.path.dirname(__file__) or ".", "hal9000.wav")
        if os.path.isfile(hal_wav):
            tts_voice = "hal9000.wav"

    # Write minimal .env
    lines = [
        "# HAL Answering Service — config (auto-generated)",
        "HOST=127.0.0.1",
        "PORT=8080",
        "",
        f"LLM_PROVIDER={llm_provider}",
        f"LLM_BASE_URL={llm_url}",
        f"LLM_API_KEY={llm_key}",
        "STT_DEVICE=auto",
        "STT_COMPUTE_TYPE=auto",
        "TTS_DEVICE=auto",
    ]
    if tts_model_dir:
        lines.append(f"TTS_MODEL_DIR={tts_model_dir}")
    if tts_voice:
        lines.append(f"TTS_VOICE_PROMPT={tts_voice}")
    if llm_model:
        lines.append(f"LLM_MODEL={llm_model}")
    lines.append(f"OWNER_NAME={owner_name}")
    lines.append("")  # trailing newline

    with open(env_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n  Settings saved to {env_path}\n")

    # Put them in the environment immediately
    os.environ["LLM_PROVIDER"] = llm_provider
    os.environ["LLM_BASE_URL"] = llm_url
    os.environ["LLM_API_KEY"] = llm_key
    if tts_model_dir:
        os.environ["TTS_MODEL_DIR"] = tts_model_dir
    if tts_voice:
        os.environ["TTS_VOICE_PROMPT"] = tts_voice
    if llm_model:
        os.environ["LLM_MODEL"] = llm_model
    os.environ["OWNER_NAME"] = owner_name
    # Stash demo owner name so it takes priority over .env OWNER_NAME
    os.environ["_DEMO_OWNER_NAME"] = owner_name


def _preflight_demo():
    """Lightweight preflight for demo mode — skip SignalWire and PUBLIC_HOST checks."""
    errors = []

    # Check Python version
    if sys.version_info < (3, 12):
        errors.append(f"Python 3.12+ required, but you have {sys.version_info.major}.{sys.version_info.minor}.")

    # Check python-dotenv
    try:
        from dotenv import load_dotenv  # noqa: F401
    except ImportError:
        errors.append(
            "python-dotenv is not installed. Run the setup script: "
            "setup.bat (Windows) or ./setup.sh (Linux/macOS)"
        )

    # Check PyTorch
    try:
        import torch  # noqa: F401
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

    if errors:
        print("\n" + "=" * 60)
        print("  SETUP ISSUES DETECTED")
        print("=" * 60)
        for i, err in enumerate(errors, 1):
            print(f"\n  {i}. {err}")
        print("\n" + "=" * 60)
        print("  Fix the above and try again.")
        print("=" * 60 + "\n")
        sys.exit(1)


def _has_signalwire_creds() -> bool:
    """Check if SignalWire credentials are configured (not placeholders)."""
    for key in ("SIGNALWIRE_PROJECT_ID", "SIGNALWIRE_TOKEN", "SIGNALWIRE_SPACE"):
        val = os.environ.get(key, "").strip()
        if not val or val.startswith("your-"):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="HAL Answering Service")
    parser.add_argument("--demo", action="store_true",
                        help="Start in demo mode (browser mic, no SignalWire needed)")
    parser.add_argument("--host", type=str, default=None,
                        help="Bind address (default: 0.0.0.0 for --demo, else from HOST env)")
    args = parser.parse_args()

    # Load .env early so auto-detection can check SignalWire creds
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Auto-detect demo mode: if no --demo flag, check for SignalWire creds
    is_demo = args.demo or not _has_signalwire_creds()

    if is_demo and not args.demo:
        log.info("No SignalWire credentials found — starting in demo mode automatically")

    if is_demo:
        _preflight_demo()
        _demo_interactive_setup()
    else:
        _preflight()

    import uvicorn
    from config import Config
    from audio import load_silero_model
    from stt import SpeechToText
    from tts import TTS
    from server import create_app

    try:
        print(BANNER_DEMO if is_demo else BANNER)
    except UnicodeEncodeError:
        print("  HAL Answering Service" + (" -- DEMO MODE" if is_demo else ""))
    t_start = time.perf_counter()
    log.info("Starting up%s...", " (demo mode)" if is_demo else "")

    config = Config()

    if is_demo:
        config.demo_mode = True
        # Use the name from interactive setup, or default to "Dave"
        demo_owner = os.environ.get("_DEMO_OWNER_NAME", "").strip()
        if demo_owner:
            config.owner_name = demo_owner
        elif not config.owner_name or config.owner_name in ("YourName", ""):
            config.owner_name = "Dave"
        # Default to all-interfaces so the demo is accessible from LAN
        if args.host is None and config.host == "127.0.0.1":
            config.host = "0.0.0.0"

    if args.host is not None:
        config.host = args.host

    log.info("Config loaded (LLM: %s, STT: %s)", config.llm_model, config.stt_model)

    import torch
    stt_device, stt_compute_type, tts_device, runtime_warnings = _resolve_runtime(
        config.stt_device, config.stt_compute_type, config.tts_device, torch.cuda.is_available()
    )
    for warning_msg in runtime_warnings:
        log.warning(warning_msg)
    log.info("Runtime devices: STT=%s (%s), TTS=%s", stt_device, stt_compute_type, tts_device)

    # Load STT
    t0 = time.perf_counter()
    stt = SpeechToText(
        model_size=config.stt_model,
        device=stt_device,
        compute_type=stt_compute_type,
        language=config.stt_language,
        beam_size=config.stt_beam_size,
        best_of=config.stt_best_of,
        no_speech_threshold=config.stt_no_speech_threshold,
        log_prob_threshold=config.stt_log_prob_threshold,
        condition_on_previous_text=config.stt_condition_on_previous_text,
        initial_prompt=config.stt_initial_prompt,
    )
    stt.load()
    log.info("STT loaded in %.1fs", time.perf_counter() - t0)

    # Load TTS (kept hot in memory)
    t0 = time.perf_counter()
    voice_prompt = config.tts_voice_prompt or None
    model_dir = config.tts_model_dir or None
    tts = TTS(voice_prompt=voice_prompt, device=tts_device, model_dir=model_dir)
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
        try:
            mulaw = b"".join(c["mulaw"] for c in tts.synthesize_mulaw_streaming(text))
            greeting_cache[period] = {"text": text, "mulaw": mulaw}
            log.info("Pre-recorded greeting (%s): %.1fs", period, len(mulaw) / 8000)
        except Exception as e:
            log.warning("Failed to pre-record greeting (%s): %s — will use live TTS", period, e)
    log.info("Greetings pre-recorded in %.1fs (%d/%d cached)",
             time.perf_counter() - t0, len(greeting_cache), 3)

    # Pre-record silence prompts
    t0 = time.perf_counter()
    silence_prompt_cache = []
    for prompt_text in SILENCE_PROMPTS:
        try:
            mulaw = b"".join(c["mulaw"] for c in tts.synthesize_mulaw_streaming(prompt_text))
            silence_prompt_cache.append({"text": prompt_text, "mulaw": mulaw})
        except Exception as e:
            log.warning("Failed to pre-record silence prompt: %s", e)
    log.info("Silence prompts pre-recorded in %.1fs (%d/%d cached)",
             time.perf_counter() - t0, len(silence_prompt_cache), len(SILENCE_PROMPTS))

    os.makedirs(config.recordings_dir, exist_ok=True)
    os.makedirs(config.metadata_dir, exist_ok=True)

    app = create_app(config, stt, tts, vad_model, greeting_cache, silence_prompt_cache)

    total = time.perf_counter() - t_start
    if is_demo:
        log.info("=" * 55)
        log.info("  DEMO READY: http://localhost:%d/demo", config.port)
        log.info("  (Use localhost for mic access)")
        log.info("=" * 55)
        if config.host == "0.0.0.0":
            log.info("Tip: for remote access on your phone, use ngrok or cloudflared:")
            log.info("     ngrok http %d", config.port)
    else:
        log.info("Ready in %.1fs — listening on %s:%d", total, config.host, config.port)

    uvicorn.run(app, host=config.host, port=config.port, log_level="warning",
                ws_max_size=65536)


if __name__ == "__main__":
    main()
