"""
SuperCaller Configuration
=========================
Dataclass-based config loaded from environment variables.
"""

import os
from dataclasses import dataclass, field


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int = 0) -> int:
    return int(os.environ.get(key, str(default)))


def _env_float(key: str, default: float = 0.0) -> float:
    return float(os.environ.get(key, str(default)))


@dataclass
class Config:
    # SignalWire
    signalwire_project_id: str = field(default_factory=lambda: _env("SIGNALWIRE_PROJECT_ID"))
    signalwire_token: str = field(default_factory=lambda: _env("SIGNALWIRE_TOKEN"))
    signalwire_space: str = field(default_factory=lambda: _env("SIGNALWIRE_SPACE"))
    signalwire_phone_number: str = field(default_factory=lambda: _env("SIGNALWIRE_PHONE_NUMBER"))
    signalwire_signing_key: str = field(default_factory=lambda: _env("SIGNALWIRE_SIGNING_KEY", ""))

    # Server
    host: str = field(default_factory=lambda: _env("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("PORT", 8080))
    public_host: str = field(default_factory=lambda: _env("PUBLIC_HOST", ""))

    # STT
    stt_model: str = field(default_factory=lambda: _env("STT_MODEL", "large-v3-turbo"))
    stt_device: str = field(default_factory=lambda: _env("STT_DEVICE", "cuda"))
    stt_compute_type: str = field(default_factory=lambda: _env("STT_COMPUTE_TYPE", "float16"))

    # LLM
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "http://127.0.0.1:1234/v1"))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", "lm-studio"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "zai-org/glm-4.7-flash"))
    llm_max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 200))
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.7))

    # TTS (Chatterbox Turbo)
    # Path to a WAV file (>5s) for voice cloning, or empty for default voice
    tts_voice_prompt: str = field(default_factory=lambda: _env("TTS_VOICE_PROMPT", ""))

    # VAD (Silero)
    vad_silence_threshold_ms: int = field(default_factory=lambda: _env_int("VAD_SILENCE_THRESHOLD_MS", 400))
    vad_speech_threshold: float = field(default_factory=lambda: _env_float("VAD_SPEECH_THRESHOLD", 0.5))
    vad_min_speech_ms: int = field(default_factory=lambda: _env_int("VAD_MIN_SPEECH_MS", 250))

    # Security
    ws_auth_token: str = field(default_factory=lambda: _env("WS_AUTH_TOKEN", ""))
    max_concurrent_calls: int = field(default_factory=lambda: _env_int("MAX_CONCURRENT_CALLS", 3))
    max_call_duration_s: int = field(default_factory=lambda: _env_int("MAX_CALL_DURATION_S", 600))

    # Recording
    recordings_dir: str = field(default_factory=lambda: _env("RECORDINGS_DIR", "recordings"))

    # Notifications
    ntfy_topic: str = field(default_factory=lambda: _env("NTFY_TOPIC", ""))

    # Owner info (for greetings)
    owner_name: str = field(default_factory=lambda: _env("OWNER_NAME", ""))
