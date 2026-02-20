"""Dataclass-based config loaded from environment variables."""

import logging
import os
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int = 0) -> int:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("Invalid integer for %s=%r, using default %d", key, raw, default)
        return default


def _env_float(key: str, default: float = 0.0) -> float:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning("Invalid float for %s=%r, using default %s", key, raw, default)
        return default


def _default_voice_prompt() -> str:
    """Return path to hal9000.wav if it exists next to this file, else empty."""
    candidate = os.path.join(os.path.dirname(__file__) or ".", "hal9000.wav")
    return candidate if os.path.isfile(candidate) else ""


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.environ.get(key, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    log.warning("Invalid boolean for %s=%r, using default %s", key, raw, default)
    return default


@dataclass
class Config:
    # SignalWire
    signalwire_project_id: str = field(default_factory=lambda: _env("SIGNALWIRE_PROJECT_ID"))
    signalwire_token: str = field(default_factory=lambda: _env("SIGNALWIRE_TOKEN"))
    signalwire_space: str = field(default_factory=lambda: _env("SIGNALWIRE_SPACE"))
    signalwire_phone_number: str = field(default_factory=lambda: _env("SIGNALWIRE_PHONE_NUMBER"))
    signalwire_signing_key: str = field(default_factory=lambda: _env("SIGNALWIRE_SIGNING_KEY", ""))

    # Server
    host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("PORT", 8080))
    public_host: str = field(default_factory=lambda: _env("PUBLIC_HOST", ""))

    # STT
    stt_model: str = field(default_factory=lambda: _env("STT_MODEL", "large-v3-turbo"))
    stt_device: str = field(default_factory=lambda: _env("STT_DEVICE", "auto"))
    stt_compute_type: str = field(default_factory=lambda: _env("STT_COMPUTE_TYPE", "auto"))
    stt_language: str = field(default_factory=lambda: _env("STT_LANGUAGE", "en"))
    stt_beam_size: int = field(default_factory=lambda: _env_int("STT_BEAM_SIZE", 1))
    stt_best_of: int = field(default_factory=lambda: _env_int("STT_BEST_OF", 1))
    stt_no_speech_threshold: float = field(default_factory=lambda: _env_float("STT_NO_SPEECH_THRESHOLD", 0.6))
    stt_log_prob_threshold: float = field(default_factory=lambda: _env_float("STT_LOG_PROB_THRESHOLD", -1.0))
    stt_condition_on_previous_text: bool = field(
        default_factory=lambda: _env_bool("STT_CONDITION_ON_PREVIOUS_TEXT", False)
    )
    stt_initial_prompt: str = field(
        default_factory=lambda: _env("STT_INITIAL_PROMPT", "Phone call screening conversation.")
    )

    # LLM
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "auto"))
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "http://127.0.0.1:1234/v1"))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", "lm-studio"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "zai-org/glm-4.7-flash"))
    llm_max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 200))
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.7))
    llm_frequency_penalty: float = field(default_factory=lambda: _env_float("LLM_FREQUENCY_PENALTY", 0.0))

    # TTS (Chatterbox Turbo)
    tts_device: str = field(default_factory=lambda: _env("TTS_DEVICE", "auto"))
    # Optional local directory containing pre-downloaded Chatterbox weights.
    # If set and valid, startup uses this directory and skips HF downloads.
    tts_model_dir: str = field(default_factory=lambda: _env("TTS_MODEL_DIR", ""))
    # Path to a WAV file (>5s) for voice cloning. Defaults to hal9000.wav if present.
    tts_voice_prompt: str = field(default_factory=lambda: _env("TTS_VOICE_PROMPT", "") or _default_voice_prompt())

    # VAD (Silero)
    vad_silence_threshold_ms: int = field(default_factory=lambda: _env_int("VAD_SILENCE_THRESHOLD_MS", 400))
    vad_speech_threshold: float = field(default_factory=lambda: _env_float("VAD_SPEECH_THRESHOLD", 0.5))
    vad_min_speech_ms: int = field(default_factory=lambda: _env_int("VAD_MIN_SPEECH_MS", 250))

    # Security
    max_concurrent_calls: int = field(default_factory=lambda: _env_int("MAX_CONCURRENT_CALLS", 3))
    max_call_duration_s: int = field(default_factory=lambda: _env_int("MAX_CALL_DURATION_S", 600))

    # Recording
    recordings_dir: str = field(default_factory=lambda: _env("RECORDINGS_DIR", "recordings"))

    # Metadata (stored separately from recordings — not web-accessible)
    metadata_dir: str = field(default_factory=lambda: _env("METADATA_DIR", "metadata"))

    # Notifications
    ntfy_topic: str = field(default_factory=lambda: _env("NTFY_TOPIC", ""))
    ntfy_token: str = field(default_factory=lambda: _env("NTFY_TOKEN", ""))

    # Owner info (for greetings)
    owner_name: str = field(default_factory=lambda: _env("OWNER_NAME", ""))

    # Demo mode (set programmatically by --demo flag, not from env)
    demo_mode: bool = False

    def __post_init__(self):
        """Validate configuration values after initialization."""
        errors = []

        # Port range
        if not (1 <= self.port <= 65535):
            errors.append(f"PORT must be 1-65535, got {self.port}")

        # Numeric ranges
        if self.llm_max_tokens < 1:
            errors.append(f"LLM_MAX_TOKENS must be >= 1, got {self.llm_max_tokens}")
        if not (0.0 <= self.llm_temperature <= 2.0):
            errors.append(f"LLM_TEMPERATURE must be 0.0-2.0, got {self.llm_temperature}")
        if not (-2.0 <= self.llm_frequency_penalty <= 2.0):
            errors.append(f"LLM_FREQUENCY_PENALTY must be -2.0-2.0, got {self.llm_frequency_penalty}")
        valid_llm_providers = {"auto", "lmstudio", "ollama", "openai_compatible"}
        if self.llm_provider.strip().lower() not in valid_llm_providers:
            errors.append(
                "LLM_PROVIDER must be one of auto, lmstudio, ollama, openai_compatible, "
                f"got {self.llm_provider!r}"
            )
        if not (0.0 < self.vad_speech_threshold <= 1.0):
            errors.append(f"VAD_SPEECH_THRESHOLD must be 0.0-1.0, got {self.vad_speech_threshold}")
        if self.stt_beam_size < 1:
            errors.append(f"STT_BEAM_SIZE must be >= 1, got {self.stt_beam_size}")
        if self.stt_best_of < 1:
            errors.append(f"STT_BEST_OF must be >= 1, got {self.stt_best_of}")
        if not (0.0 <= self.stt_no_speech_threshold <= 1.0):
            errors.append(
                "STT_NO_SPEECH_THRESHOLD must be 0.0-1.0, "
                f"got {self.stt_no_speech_threshold}"
            )
        if self.vad_silence_threshold_ms < 50:
            errors.append(f"VAD_SILENCE_THRESHOLD_MS must be >= 50, got {self.vad_silence_threshold_ms}")
        if self.vad_min_speech_ms < 0:
            errors.append(f"VAD_MIN_SPEECH_MS must be >= 0, got {self.vad_min_speech_ms}")
        if self.max_concurrent_calls < 1:
            errors.append(f"MAX_CONCURRENT_CALLS must be >= 1, got {self.max_concurrent_calls}")
        if self.max_call_duration_s < 10:
            errors.append(f"MAX_CALL_DURATION_S must be >= 10, got {self.max_call_duration_s}")

        # Voice prompt file check
        if self.tts_voice_prompt and not os.path.isfile(self.tts_voice_prompt):
            errors.append(f"TTS_VOICE_PROMPT file not found: {self.tts_voice_prompt}")
        if self.tts_model_dir and not os.path.isdir(self.tts_model_dir):
            errors.append(f"TTS_MODEL_DIR directory not found: {self.tts_model_dir}")

        if errors:
            raise ValueError("Configuration errors:\n  " + "\n  ".join(errors))

