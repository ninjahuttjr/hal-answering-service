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
    """Check for dependency / install issues.  Exits on fatal errors only."""
    errors = []

    try:
        from dotenv import load_dotenv  # noqa: F401
    except ImportError:
        errors.append(
            "python-dotenv is not installed. Run the setup script: "
            "setup.bat (Windows) or ./setup.sh (Linux/macOS)"
        )

    if sys.version_info < (3, 12):
        errors.append(f"Python 3.12+ required, but you have {sys.version_info.major}.{sys.version_info.minor}.")

    try:
        import torch  # noqa: F401
    except ImportError:
        errors.append("PyTorch is not installed. Run the setup script: setup.bat (Windows) or ./setup.sh (Linux/macOS)")

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


# ── Helpers for interactive setup ──

def _infer_llm_provider_from_url(base_url: str) -> str:
    url = (base_url or "").strip().lower()
    if "127.0.0.1:1234" in url or "localhost:1234" in url:
        return "lmstudio"
    if "127.0.0.1:11434" in url or "localhost:11434" in url or "ollama" in url:
        return "ollama"
    return "openai_compatible"


def _ask(prompt: str, default: str = "") -> str:
    """Prompt the user with an optional default shown in brackets."""
    if default:
        answer = input(f"  {prompt} [{default}]: ").strip()
    else:
        answer = input(f"  {prompt}: ").strip()
    return answer if answer else default


def _is_placeholder(val: str) -> bool:
    """Return True if *val* looks like an unfilled placeholder."""
    return (
        not val
        or val.startswith("your-")
        or val in ("+1XXXXXXXXXX", "YourName")
    )


def _is_production_ready() -> bool:
    """Return True when every field required for live calls is filled in."""
    for key in ("SIGNALWIRE_PROJECT_ID", "SIGNALWIRE_TOKEN", "SIGNALWIRE_SPACE",
                "SIGNALWIRE_PHONE_NUMBER", "PUBLIC_HOST", "OWNER_NAME"):
        if _is_placeholder(os.environ.get(key, "").strip()):
            return False
    return True


def _missing_prod_fields() -> list[tuple[str, str]]:
    """Return [(KEY, description), ...] for every production field still empty."""
    checks = [
        ("SIGNALWIRE_PROJECT_ID", "SignalWire project ID"),
        ("SIGNALWIRE_TOKEN",      "SignalWire API token"),
        ("SIGNALWIRE_SPACE",      "SignalWire space name"),
        ("SIGNALWIRE_PHONE_NUMBER", "SignalWire phone number"),
        ("SIGNALWIRE_SIGNING_KEY", "SignalWire signing key (webhook security)"),
        ("PUBLIC_HOST",           "Public hostname (e.g. caller.example.com)"),
        ("OWNER_NAME",            "Your name (used in greetings)"),
    ]
    return [(k, d) for k, d in checks if _is_placeholder(os.environ.get(k, "").strip())]


# ── Unified interactive setup ──

def _interactive_setup():
    """Walk the user through first-time configuration.

    * Always writes a **complete** .env (every field present).
    * Optional fields are included as comments showing their defaults.
    * If the user skips the SignalWire / production section, HAL will
      start in demo mode automatically — no need for a separate flow.
    """
    env_path = os.path.join(os.path.dirname(__file__) or ".", ".env")
    already_has_env = os.path.exists(env_path)

    # Load existing values so we can use them as defaults
    if already_has_env:
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path, override=False)
        except ImportError:
            pass

    # Determine if setup is needed at all
    llm_url = os.environ.get("LLM_BASE_URL", "").strip()
    owner   = os.environ.get("OWNER_NAME", "").strip()
    if already_has_env and llm_url and owner and not _is_placeholder(owner):
        return  # Already configured — nothing to do

    # ── env helper: read existing or return empty ──
    def _cur(key: str) -> str:
        return os.environ.get(key, "").strip()

    # ────────────────────────────────────────────────
    #  Welcome
    # ────────────────────────────────────────────────
    print()
    print("  " + "=" * 54)
    print("   HAL Answering Service — First-Time Setup")
    print("  " + "=" * 54)
    print()
    print("  This wizard will create your .env configuration file.")
    print("  Press Enter to accept the default shown in [brackets].")
    print("  You can always edit .env later to change any value.")
    print()

    # ────────────────────────────────────────────────
    #  1. Owner / Identity
    # ────────────────────────────────────────────────
    print("  --- Your Info ---")
    print("  HAL greets callers with \"You've reached <name>'s phone.\"")
    owner_name = _ask("Your name", _cur("OWNER_NAME") or "Dave")
    print()

    # ────────────────────────────────────────────────
    #  2. LLM Provider
    # ────────────────────────────────────────────────
    print("  --- LLM (Language Model) ---")
    print("  HAL needs a local LLM server for conversations.")
    print("    1) LM Studio      (default: http://127.0.0.1:1234/v1)")
    print("    2) Ollama          (default: http://127.0.0.1:11434/v1)")
    print("    3) Other OpenAI-compatible server")

    cur_provider = _cur("LLM_PROVIDER")
    default_choice = "1"
    if cur_provider == "ollama":
        default_choice = "2"
    elif cur_provider == "openai_compatible":
        default_choice = "3"
    elif _cur("LLM_BASE_URL"):
        inferred = _infer_llm_provider_from_url(_cur("LLM_BASE_URL"))
        default_choice = {"lmstudio": "1", "ollama": "2"}.get(inferred, "3")

    provider_choice = _ask("Choose provider", default_choice)
    if provider_choice == "2":
        llm_provider = "ollama"
    elif provider_choice == "3":
        llm_provider = "openai_compatible"
    else:
        llm_provider = "lmstudio"

    # Provider-specific defaults
    if llm_provider == "lmstudio":
        def_url = _cur("LLM_BASE_URL") or "http://127.0.0.1:1234/v1"
        def_key = _cur("LLM_API_KEY") or "lm-studio"
        def_model = _cur("LLM_MODEL") or ""
        model_hint = "LLM model name (blank = server default)"
    elif llm_provider == "ollama":
        def_url = _cur("LLM_BASE_URL") or "http://127.0.0.1:11434/v1"
        def_key = _cur("LLM_API_KEY") or "ollama"
        def_model = _cur("LLM_MODEL") or "qwen3:4b"
        model_hint = "Ollama model (must match `ollama list`)"
    else:
        def_url = _cur("LLM_BASE_URL") or "http://127.0.0.1:1234/v1"
        def_key = _cur("LLM_API_KEY") or "local"
        def_model = _cur("LLM_MODEL") or ""
        model_hint = "LLM model name"

    llm_url   = _ask("LLM server URL", def_url)
    llm_key   = _ask("LLM API key", def_key)
    llm_model = _ask(model_hint, def_model)
    print()

    # ────────────────────────────────────────────────
    #  3. SignalWire (production phone line)
    # ────────────────────────────────────────────────
    print("  --- SignalWire (Phone Line) ---")
    print("  To answer real phone calls, HAL needs a SignalWire account.")
    print("  Leave these blank to start in demo mode (browser mic only).")
    print("  You can fill them in later by editing .env.\n")

    sw_project = _ask("SignalWire Project ID",    _cur("SIGNALWIRE_PROJECT_ID") if not _is_placeholder(_cur("SIGNALWIRE_PROJECT_ID")) else "")
    sw_token   = _ask("SignalWire API Token",      _cur("SIGNALWIRE_TOKEN")      if not _is_placeholder(_cur("SIGNALWIRE_TOKEN"))      else "")
    sw_space   = _ask("SignalWire Space name",     _cur("SIGNALWIRE_SPACE")      if not _is_placeholder(_cur("SIGNALWIRE_SPACE"))      else "")
    sw_phone   = _ask("SignalWire Phone Number (e.g. +14155551234)",
                       _cur("SIGNALWIRE_PHONE_NUMBER") if not _is_placeholder(_cur("SIGNALWIRE_PHONE_NUMBER")) else "")
    sw_signing = _ask("SignalWire Signing Key (blank = use API token)",
                       _cur("SIGNALWIRE_SIGNING_KEY") if not _is_placeholder(_cur("SIGNALWIRE_SIGNING_KEY")) else "")
    print()

    # ────────────────────────────────────────────────
    #  4. Public hostname / tunnel
    # ────────────────────────────────────────────────
    print("  --- Public Hostname ---")
    print("  SignalWire sends webhooks to this address.  Just the hostname,")
    print("  no https:// prefix.  Example: caller.example.com")
    print("  (Leave blank if you don't have one yet — demo mode still works.)\n")
    public_host = _ask("Public hostname",
                        _cur("PUBLIC_HOST") if not _is_placeholder(_cur("PUBLIC_HOST")) else "")
    print()

    # ────────────────────────────────────────────────
    #  5. Server settings
    # ────────────────────────────────────────────────
    print("  --- Server ---")
    host = _ask("Bind address (127.0.0.1 = local only, 0.0.0.0 = all interfaces)",
                 _cur("HOST") or "127.0.0.1")
    port = _ask("Port", _cur("PORT") or "8080")

    # NO_TLS (helpful hint)
    cur_no_tls = _cur("NO_TLS")
    if not cur_no_tls and public_host:
        # If they have a tunnel hostname, they probably want NO_TLS
        print("\n  Tip: If you're behind Cloudflare Tunnel, ngrok, etc., the")
        print("  tunnel already handles TLS. Disable local TLS to avoid 502 errors.")
    no_tls = _ask("Disable local TLS? (yes if behind a tunnel)", cur_no_tls or ("yes" if public_host else "no"))
    no_tls_val = "1" if no_tls.lower() in ("1", "true", "yes", "y") else ""
    print()

    # ────────────────────────────────────────────────
    #  Auto-detect TTS voice / model dir (no prompt)
    # ────────────────────────────────────────────────
    tts_voice = _cur("TTS_VOICE_PROMPT")
    if not tts_voice:
        hal_wav = os.path.join(os.path.dirname(__file__) or ".", "hal9000.wav")
        if os.path.isfile(hal_wav):
            tts_voice = "hal9000.wav"

    tts_model_dir = _cur("TTS_MODEL_DIR")
    bundled = os.path.join(os.path.dirname(__file__) or ".", "models", "chatterbox")
    if not tts_model_dir and os.path.isdir(bundled):
        tts_model_dir = bundled

    # ────────────────────────────────────────────────
    #  Write COMPLETE .env
    # ────────────────────────────────────────────────
    env_lines = [
        "# ══════════════════════════════════════════════════════════",
        "# HAL Answering Service — Configuration",
        "# ══════════════════════════════════════════════════════════",
        "# Generated by the setup wizard.  Edit any time.",
        "# Lines starting with # are comments / optional defaults.",
        "",
        "# --- Owner ---",
        f"OWNER_NAME={owner_name}",
        "",
        "# --- LLM (Language Model) ---",
        f"LLM_PROVIDER={llm_provider}",
        f"LLM_BASE_URL={llm_url}",
        f"LLM_API_KEY={llm_key}",
        f"LLM_MODEL={llm_model}" if llm_model else "# LLM_MODEL=                # blank = server default",
        "# LLM_MAX_TOKENS=200        # max response tokens",
        "# LLM_TEMPERATURE=0.7       # 0.0-2.0",
        "# LLM_FREQUENCY_PENALTY=0.0 # -2.0 to 2.0",
        "",
        "# --- SignalWire (required for live phone calls) ---",
        "# Get credentials at: https://signalwire.com",
    ]
    if sw_project:
        env_lines.append(f"SIGNALWIRE_PROJECT_ID={sw_project}")
    else:
        env_lines.append("# SIGNALWIRE_PROJECT_ID=    # your project UUID")
    if sw_token:
        env_lines.append(f"SIGNALWIRE_TOKEN={sw_token}")
    else:
        env_lines.append("# SIGNALWIRE_TOKEN=         # your API token")
    if sw_space:
        env_lines.append(f"SIGNALWIRE_SPACE={sw_space}")
    else:
        env_lines.append("# SIGNALWIRE_SPACE=         # your space name")
    if sw_phone:
        env_lines.append(f"SIGNALWIRE_PHONE_NUMBER={sw_phone}")
    else:
        env_lines.append("# SIGNALWIRE_PHONE_NUMBER=  # e.g. +14155551234")
    if sw_signing:
        env_lines.append(f"SIGNALWIRE_SIGNING_KEY={sw_signing}")
    else:
        env_lines.append("# SIGNALWIRE_SIGNING_KEY=   # blank = falls back to SIGNALWIRE_TOKEN")

    env_lines += [
        "",
        "# --- Server ---",
        f"HOST={host}",
        f"PORT={port}",
    ]
    if public_host:
        env_lines.append(f"PUBLIC_HOST={public_host}")
    else:
        env_lines.append("# PUBLIC_HOST=              # e.g. caller.example.com (no https://)")
    if no_tls_val:
        env_lines.append(f"NO_TLS={no_tls_val}")
    else:
        env_lines.append("# NO_TLS=                   # set to 1 if behind a reverse proxy / tunnel")

    env_lines += [
        "",
        "# --- Voice / TTS (Chatterbox) ---",
        "# Device: auto, cuda, or cpu",
        "TTS_DEVICE=auto",
    ]
    if tts_voice:
        env_lines.append(f"TTS_VOICE_PROMPT={tts_voice}")
    else:
        env_lines.append("# TTS_VOICE_PROMPT=hal9000.wav  # WAV file (>5s) for voice cloning")
    if tts_model_dir:
        env_lines.append(f"TTS_MODEL_DIR={tts_model_dir}")
    else:
        env_lines.append("# TTS_MODEL_DIR=            # local Chatterbox weights (skips download)")
    env_lines.append("# HF_TOKEN=                 # Hugging Face token if anonymous download fails")

    env_lines += [
        "",
        "# --- Speech-to-Text / Faster-Whisper ---",
        "# STT_MODEL=large-v3-turbo  # model size (tiny/base/small/medium/large-v3-turbo)",
        "STT_DEVICE=auto",
        "STT_COMPUTE_TYPE=auto",
        "# STT_LANGUAGE=en           # blank = auto-detect",
        "# STT_BEAM_SIZE=1",
        "# STT_BEST_OF=1",
        "# STT_NO_SPEECH_THRESHOLD=0.6",
        "# STT_LOG_PROB_THRESHOLD=-1.0",
        "# STT_CONDITION_ON_PREVIOUS_TEXT=false",
        "# STT_INITIAL_PROMPT=Phone call screening conversation.",
        "",
        "# --- Voice Activity Detection (Silero) ---",
        "# VAD_SPEECH_THRESHOLD=0.5",
        "# VAD_SILENCE_THRESHOLD_MS=400",
        "# VAD_MIN_SPEECH_MS=250",
        "",
        "# --- Security ---",
        "# MAX_CONCURRENT_CALLS=3    # simultaneous calls",
        "# MAX_CALL_DURATION_S=600   # 10-minute max per call",
        "",
        "# --- Recording & Metadata ---",
        "# RECORDINGS_DIR=recordings",
        "# METADATA_DIR=metadata",
        "",
        "# --- Push Notifications (ntfy.sh, optional) ---",
        "# NTFY_TOPIC=               # create a topic at https://ntfy.sh",
        "# NTFY_TOKEN=               # auth token for private topics",
        "",
    ]

    with open(env_path, "w", encoding="utf-8") as f:
        f.write("\n".join(env_lines))

    # Push all values into the environment immediately
    env_map = {
        "OWNER_NAME": owner_name,
        "LLM_PROVIDER": llm_provider,
        "LLM_BASE_URL": llm_url,
        "LLM_API_KEY": llm_key,
        "HOST": host,
        "PORT": port,
    }
    if llm_model:
        env_map["LLM_MODEL"] = llm_model
    if sw_project:
        env_map["SIGNALWIRE_PROJECT_ID"] = sw_project
    if sw_token:
        env_map["SIGNALWIRE_TOKEN"] = sw_token
    if sw_space:
        env_map["SIGNALWIRE_SPACE"] = sw_space
    if sw_phone:
        env_map["SIGNALWIRE_PHONE_NUMBER"] = sw_phone
    if sw_signing:
        env_map["SIGNALWIRE_SIGNING_KEY"] = sw_signing
    if public_host:
        env_map["PUBLIC_HOST"] = public_host
    if no_tls_val:
        env_map["NO_TLS"] = no_tls_val
    if tts_voice:
        env_map["TTS_VOICE_PROMPT"] = tts_voice
    if tts_model_dir:
        env_map["TTS_MODEL_DIR"] = tts_model_dir
    for k, v in env_map.items():
        os.environ[k] = v

    # ── Summary ──
    print("  " + "=" * 54)
    print(f"  Config saved to {env_path}")
    print("  " + "=" * 54)
    has_sw = all([sw_project, sw_token, sw_space, sw_phone])
    if has_sw and public_host and owner_name:
        print("  Mode: PRODUCTION (live phone calls)")
    else:
        print("  Mode: DEMO (browser mic)")
        missing = []
        if not has_sw:
            missing.append("SignalWire credentials")
        if not public_host:
            missing.append("PUBLIC_HOST")
        if not owner_name:
            missing.append("OWNER_NAME")
        if missing:
            print(f"  To enable live calls, fill in: {', '.join(missing)}")
            print("  Just edit .env and restart — no re-setup needed.")
    print()


def _resolve_ssl(config, args) -> tuple:
    """Return (certfile, keyfile, hostname) or (None, None, None).

    Priority order:
      1. --ssl-cert / --ssl-key CLI args
      2. SSL_CERTFILE / SSL_KEYFILE env vars
      3. Auto-generated Tailscale cert (if `tailscale cert` is available)
    """
    import shutil
    import subprocess

    # 0. Explicit disable (useful behind Cloudflare Tunnel / reverse proxy)
    if os.environ.get("NO_TLS", "").strip().lower() in ("1", "true", "yes"):
        return None, None, None

    # 1. Explicit CLI / env
    certfile = getattr(args, "ssl_cert", None) or os.environ.get("SSL_CERTFILE", "").strip()
    keyfile = getattr(args, "ssl_key", None) or os.environ.get("SSL_KEYFILE", "").strip()
    if certfile and keyfile and os.path.isfile(certfile) and os.path.isfile(keyfile):
        return certfile, keyfile, None

    # 2. Auto Tailscale
    if shutil.which("tailscale"):
        try:
            result = subprocess.run(
                ["tailscale", "status", "--json"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                import json
                ts = json.loads(result.stdout)
                dns_name = ts.get("Self", {}).get("DNSName", "").rstrip(".")
                if dns_name:
                    base = os.path.dirname(__file__) or "."
                    ts_cert = os.path.join(base, "tailscale.crt")
                    ts_key = os.path.join(base, "tailscale.key")
                    gen = subprocess.run(
                        ["tailscale", "cert",
                         "--cert-file", ts_cert, "--key-file", ts_key,
                         dns_name],
                        capture_output=True, text=True, timeout=15
                    )
                    if gen.returncode == 0 and os.path.isfile(ts_cert):
                        log.info("Tailscale HTTPS cert for %s", dns_name)
                        return ts_cert, ts_key, dns_name
        except Exception as e:
            log.debug("Tailscale cert auto-detection failed: %s", e)

    return None, None, None


def main():
    parser = argparse.ArgumentParser(description="HAL Answering Service")
    parser.add_argument("--demo", action="store_true",
                        help="Force demo mode (browser mic, no SignalWire needed)")
    parser.add_argument("--host", type=str, default=None,
                        help="Bind address (default: 0.0.0.0 for demo, else from HOST env)")
    parser.add_argument("--ssl-cert", type=str, default=None,
                        help="Path to SSL certificate file (also: SSL_CERTFILE env)")
    parser.add_argument("--ssl-key", type=str, default=None,
                        help="Path to SSL private key file (also: SSL_KEYFILE env)")
    args = parser.parse_args()

    # Load .env early so auto-detection can check existing values
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # ── Dependency check (fatal errors only) ──
    _preflight()

    # ── Interactive setup (first run, or .env incomplete) ──
    _interactive_setup()

    # ── Re-load .env in case setup just wrote / rewrote it ──
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except ImportError:
        pass

    # ── Decide mode: demo unless ALL production fields are present ──
    is_demo = args.demo or not _is_production_ready()

    if is_demo and not args.demo:
        missing = _missing_prod_fields()
        if missing:
            log.info("Demo mode — missing fields for live calls: %s",
                     ", ".join(k for k, _ in missing))
        else:
            log.info("Starting in demo mode (--demo flag)")

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
        if not config.owner_name or _is_placeholder(config.owner_name):
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

    # ── Resolve SSL cert/key ──
    ssl_certfile, ssl_keyfile, ssl_hostname = _resolve_ssl(config, args)
    use_tls = bool(ssl_certfile and ssl_keyfile)
    scheme = "https" if use_tls else "http"

    total = time.perf_counter() - t_start
    if is_demo:
        log.info("=" * 55)
        if use_tls:
            host_display = ssl_hostname or "<your-hostname>"
            log.info("  DEMO READY: %s://%s:%d/demo", scheme, host_display, config.port)
        else:
            log.info("  DEMO READY: http://localhost:%d/demo", config.port)
            log.info("  (Use localhost for mic access)")
        log.info("=" * 55)
        if config.host == "0.0.0.0" and not use_tls:
            log.info("Tip: for remote access on your phone, use ngrok or cloudflared:")
            log.info("     ngrok http %d", config.port)
    else:
        log.info("Ready in %.1fs — listening on %s://%s:%d", total, scheme, config.host, config.port)

    ssl_kwargs = {}
    if use_tls:
        ssl_kwargs = {"ssl_certfile": ssl_certfile, "ssl_keyfile": ssl_keyfile}
        log.info("TLS enabled (%s)", ssl_certfile)

    uvicorn.run(app, host=config.host, port=config.port, log_level="warning",
                ws_max_size=65536, **ssl_kwargs)


if __name__ == "__main__":
    main()
