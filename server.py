"""FastAPI server with SignalWire webhook and WebSocket media stream endpoints."""

import asyncio
import base64
import binascii
import collections
import copy
import json
import logging
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import Response

from config import Config
from audio import SileroVAD
from stt import SpeechToText
from tts import TTS, REQUIRED_MODEL_FILES
from call_handler import CallHandler

log = logging.getLogger(__name__)

# How long a validated CallSid remains valid before being pruned (seconds)
CALLSID_TTL_S = 30

# How long a WebSocket can stay connected without sending a valid "start" event
WS_START_TIMEOUT_S = 15

# Max WebSocket frame size (SignalWire media frames are small — 20ms at 8kHz = ~160 bytes base64)
WS_MAX_FRAME_SIZE = 65536

# Max decoded audio payload size (bytes) — reject oversized base64 payloads before decode
MAX_AUDIO_PAYLOAD_B64_LEN = 32000  # ~24KB decoded, well above normal 160-byte frames

# Upper bound on validated_call_sids dict to prevent memory exhaustion
MAX_VALIDATED_SIDS = 100

# Webhook rate limiting: max requests per IP within the sliding window
WEBHOOK_RATE_LIMIT = 20           # max requests per window
WEBHOOK_RATE_WINDOW_S = 60        # sliding window in seconds
WEBHOOK_RATE_MAX_IPS = 500        # max tracked IPs (prevent memory exhaustion)


DEMO_SETTINGS_FIELDS = [
    # SignalWire
    {"env": "SIGNALWIRE_PROJECT_ID", "attr": "signalwire_project_id", "type": "str",
     "section": "SignalWire", "secret": False, "apply_runtime": False,
     "description": "SignalWire project ID (production use)."},
    {"env": "SIGNALWIRE_TOKEN", "attr": "signalwire_token", "type": "str",
     "section": "SignalWire", "secret": True, "apply_runtime": False,
     "description": "SignalWire API token (production use)."},
    {"env": "SIGNALWIRE_SPACE", "attr": "signalwire_space", "type": "str",
     "section": "SignalWire", "secret": False, "apply_runtime": False,
     "description": "SignalWire space name."},
    {"env": "SIGNALWIRE_PHONE_NUMBER", "attr": "signalwire_phone_number", "type": "str",
     "section": "SignalWire", "secret": False, "apply_runtime": False,
     "description": "Phone number used for inbound calls."},
    {"env": "SIGNALWIRE_SIGNING_KEY", "attr": "signalwire_signing_key", "type": "str",
     "section": "SignalWire", "secret": True, "apply_runtime": False,
     "description": "Webhook signature validation key."},

    # Server
    {"env": "HOST", "attr": "host", "type": "str", "section": "Server", "secret": False,
     "apply_runtime": False, "description": "Bind host (requires restart)."},
    {"env": "PORT", "attr": "port", "type": "int", "section": "Server", "secret": False,
     "apply_runtime": False, "description": "Bind port (requires restart)."},
    {"env": "PUBLIC_HOST", "attr": "public_host", "type": "str", "section": "Server",
     "secret": False, "apply_runtime": True, "description": "Public host for webhooks/streams."},
    {"env": "MAX_CONCURRENT_CALLS", "attr": "max_concurrent_calls", "type": "int",
     "section": "Server", "secret": False, "apply_runtime": True,
     "description": "Max concurrent calls."},
    {"env": "MAX_CALL_DURATION_S", "attr": "max_call_duration_s", "type": "int",
     "section": "Server", "secret": False, "apply_runtime": True,
     "description": "Max call duration in seconds."},

    # STT
    {"env": "STT_MODEL", "attr": "stt_model", "type": "str", "section": "STT", "secret": False,
     "apply_runtime": False, "description": "Whisper model size (requires restart)."},
    {"env": "STT_DEVICE", "attr": "stt_device", "type": "str", "section": "STT", "secret": False,
     "apply_runtime": False, "description": "STT runtime device (requires restart)."},
    {"env": "STT_COMPUTE_TYPE", "attr": "stt_compute_type", "type": "str", "section": "STT",
     "secret": False, "apply_runtime": False, "description": "STT precision/compute type (restart)."},
    {"env": "STT_LANGUAGE", "attr": "stt_language", "type": "str", "section": "STT",
     "secret": False, "apply_runtime": True, "description": "Language code (blank for auto-detect)."},
    {"env": "STT_BEAM_SIZE", "attr": "stt_beam_size", "type": "int", "section": "STT",
     "secret": False, "apply_runtime": True, "description": "Beam width (1 = fastest)."},
    {"env": "STT_BEST_OF", "attr": "stt_best_of", "type": "int", "section": "STT",
     "secret": False, "apply_runtime": True, "description": "Candidate samples per segment."},
    {"env": "STT_NO_SPEECH_THRESHOLD", "attr": "stt_no_speech_threshold", "type": "float", "section": "STT",
     "secret": False, "apply_runtime": True, "description": "Reject threshold for likely silence."},
    {"env": "STT_LOG_PROB_THRESHOLD", "attr": "stt_log_prob_threshold", "type": "float", "section": "STT",
     "secret": False, "apply_runtime": True, "description": "Minimum log-prob threshold."},
    {"env": "STT_CONDITION_ON_PREVIOUS_TEXT", "attr": "stt_condition_on_previous_text", "type": "bool",
     "section": "STT", "secret": False, "apply_runtime": True,
     "description": "Carry prior context between segments (accuracy vs speed)."},
    {"env": "STT_INITIAL_PROMPT", "attr": "stt_initial_prompt", "type": "str", "section": "STT",
     "secret": False, "apply_runtime": True, "description": "Priming prompt for Whisper decoding."},

    # LLM
    {"env": "LLM_PROVIDER", "attr": "llm_provider", "type": "str", "section": "LLM",
     "secret": False, "apply_runtime": True, "description": "auto, lmstudio, ollama, openai_compatible."},
    {"env": "LLM_BASE_URL", "attr": "llm_base_url", "type": "str", "section": "LLM",
     "secret": False, "apply_runtime": True, "description": "OpenAI-compatible base URL."},
    {"env": "LLM_API_KEY", "attr": "llm_api_key", "type": "str", "section": "LLM",
     "secret": True, "apply_runtime": True, "description": "API key for LLM endpoint."},
    {"env": "LLM_MODEL", "attr": "llm_model", "type": "str", "section": "LLM",
     "secret": False, "apply_runtime": True, "description": "Model name for chat completions."},
    {"env": "LLM_MAX_TOKENS", "attr": "llm_max_tokens", "type": "int", "section": "LLM",
     "secret": False, "apply_runtime": True, "description": "Max output tokens per response."},
    {"env": "LLM_TEMPERATURE", "attr": "llm_temperature", "type": "float", "section": "LLM",
     "secret": False, "apply_runtime": True, "description": "Sampling temperature."},
    {"env": "LLM_FREQUENCY_PENALTY", "attr": "llm_frequency_penalty", "type": "float", "section": "LLM",
     "secret": False, "apply_runtime": True, "description": "Repetition penalty."},

    # TTS
    {"env": "TTS_DEVICE", "attr": "tts_device", "type": "str", "section": "TTS",
     "secret": False, "apply_runtime": False, "description": "TTS runtime device (requires restart)."},
    {"env": "TTS_MODEL_DIR", "attr": "tts_model_dir", "type": "str", "section": "TTS",
     "secret": False, "apply_runtime": False, "description": "Local bundled Chatterbox model dir."},
    {"env": "TTS_VOICE_PROMPT", "attr": "tts_voice_prompt", "type": "str", "section": "TTS",
     "secret": False, "apply_runtime": False, "description": "Voice prompt WAV path (requires restart)."},
    {"env": "HF_TOKEN", "attr": None, "type": "str", "section": "TTS",
     "secret": True, "apply_runtime": False, "description": "HF auth token used by fallback downloader."},

    # VAD
    {"env": "VAD_SPEECH_THRESHOLD", "attr": "vad_speech_threshold", "type": "float", "section": "VAD",
     "secret": False, "apply_runtime": True, "description": "Speech probability threshold."},
    {"env": "VAD_SILENCE_THRESHOLD_MS", "attr": "vad_silence_threshold_ms", "type": "int", "section": "VAD",
     "secret": False, "apply_runtime": True, "description": "Silence threshold in ms."},
    {"env": "VAD_MIN_SPEECH_MS", "attr": "vad_min_speech_ms", "type": "int", "section": "VAD",
     "secret": False, "apply_runtime": True, "description": "Minimum speech duration in ms."},

    # Storage & notifications
    {"env": "RECORDINGS_DIR", "attr": "recordings_dir", "type": "str", "section": "Storage",
     "secret": False, "apply_runtime": True, "description": "Directory for recordings."},
    {"env": "METADATA_DIR", "attr": "metadata_dir", "type": "str", "section": "Storage",
     "secret": False, "apply_runtime": True, "description": "Directory for metadata JSON."},
    {"env": "NTFY_TOPIC", "attr": "ntfy_topic", "type": "str", "section": "Notifications",
     "secret": False, "apply_runtime": True, "description": "ntfy.sh topic for notifications."},
    {"env": "NTFY_TOKEN", "attr": "ntfy_token", "type": "str", "section": "Notifications",
     "secret": True, "apply_runtime": True, "description": "ntfy auth token."},

    # Owner
    {"env": "OWNER_NAME", "attr": "owner_name", "type": "str", "section": "General",
     "secret": False, "apply_runtime": True, "description": "Name used in greetings/prompts."},
]

_DEMO_FIELD_BY_ENV = {f["env"]: f for f in DEMO_SETTINGS_FIELDS}
_DEMO_STT_RUNTIME_ATTR_MAP = {
    "STT_LANGUAGE": "language",
    "STT_BEAM_SIZE": "beam_size",
    "STT_BEST_OF": "best_of",
    "STT_NO_SPEECH_THRESHOLD": "no_speech_threshold",
    "STT_LOG_PROB_THRESHOLD": "log_prob_threshold",
    "STT_CONDITION_ON_PREVIOUS_TEXT": "condition_on_previous_text",
    "STT_INITIAL_PROMPT": "initial_prompt",
}


def _typed_value(raw: str, type_name: str):
    if type_name == "int":
        return int(raw)
    if type_name == "float":
        return float(raw)
    if type_name == "bool":
        normalized = raw.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError("must be true/false")
    return raw


def _stringify_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _format_env_value(raw: str) -> str:
    value = (raw or "").replace("\r", " ").replace("\n", " ")
    if not value:
        return ""
    # Quote values containing spaces, comments, or quotes.
    if re.search(r"\s|#|\"|'|=", value):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value


def _upsert_env_file(path: Path, updates: dict[str, str]):
    lines: list[str] = []
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()

    key_re = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        match = key_re.match(line)
        if not match:
            out.append(line)
            continue
        key = match.group(1)
        if key in updates:
            out.append(f"{key}={_format_env_value(updates[key])}")
            seen.add(key)
        else:
            out.append(line)

    for key, value in updates.items():
        if key not in seen:
            out.append(f"{key}={_format_env_value(value)}")

    path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")


def _validate_webhook(config: Config, request: Request, form: dict) -> bool:
    """Validate SignalWire webhook signature. Returns True if valid."""
    try:
        from signalwire.request_validator import RequestValidator
        signing_key = config.signalwire_signing_key or config.signalwire_token
        validator = RequestValidator(signing_key)
        signature = request.headers.get("x-signalwire-signature", "")
        # Use the public URL that SignalWire actually signed against,
        # not the local URL seen behind the tunnel proxy
        public_url = f"https://{config.public_host}{request.url.path}"
        if request.url.query:
            public_url += f"?{request.url.query}"
        valid = validator.validate(public_url, form, signature)
        if not valid:
            # Also try with the raw request URL in case proxy forwards correctly
            raw_url = str(request.url)
            valid = validator.validate(raw_url, form, signature)
            if valid:
                log.warning("Webhook signature valid only via raw request URL (%s), "
                            "not public URL (%s). Check PUBLIC_HOST / proxy config.",
                            raw_url, public_url)
        return valid
    except ImportError:
        log.error("signalwire.request_validator not available — REJECTING all webhooks. "
                  "Install the signalwire package.")
        return False
    except Exception as e:
        log.error("Webhook validation error: %s", e)
        return False


def _duration_human(duration: float) -> str:
    mins, secs = divmod(max(0, int(duration)), 60)
    return f"{mins}m {secs}s" if mins else f"{secs}s"


def _safe_token(value: str, fallback: str = "unknown") -> str:
    token = "".join(ch for ch in (value or "") if ch.isalnum())
    return token or fallback


def _persist_call_metadata(
    config: Config,
    caller_number: str,
    call_sid: str,
    duration: float,
    summary: str,
    transcript: list[dict] | None = None,
    recording_path: str = "",
):
    """Persist call metadata as JSON alongside recordings."""
    try:
        meta_dir = Path(config.metadata_dir)
        meta_dir.mkdir(parents=True, exist_ok=True)

        rec_name = Path(recording_path).name if recording_path else ""

        now = datetime.now(timezone.utc)
        payload = {
            "created_at": now.isoformat(),
            "caller_number": caller_number,
            "call_sid": call_sid,
            "duration_seconds": round(max(0.0, duration), 2),
            "duration_human": _duration_human(duration),
            "summary": (summary or "").strip(),
            "transcript": transcript or [],
            "recording_file": rec_name,
        }

        fname = f"{now.strftime('%Y%m%d_%H%M%S')}_{_safe_token(call_sid[:12])}.json"
        meta_path = meta_dir / fname
        meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    except Exception as e:
        log.error("Failed to persist call metadata: %s", e)


def create_app(config: Config, stt: SpeechToText, tts: TTS, vad_model,
               greeting_cache: dict | None = None,
               silence_prompt_cache: list | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="HAL Answering Service", docs_url=None, redoc_url=None, openapi_url=None)
    recordings_dir = Path(config.recordings_dir)
    recordings_dir.mkdir(parents=True, exist_ok=True)

    # Track active calls and validated call SIDs with timestamp for TTL
    active_calls: dict[str, CallHandler] = {}
    validated_call_sids: dict[str, float] = {}  # {call_sid: timestamp}

    # Webhook rate limiting: per-IP sliding window of request timestamps
    _webhook_hits: dict[str, collections.deque] = {}

    def _prune_stale_callsids():
        """Remove validated CallSids older than TTL and enforce upper bound."""
        now = time.monotonic()
        stale = [sid for sid, ts in validated_call_sids.items()
                 if now - ts > CALLSID_TTL_S]
        for sid in stale:
            validated_call_sids.pop(sid, None)
            log.debug("Pruned stale CallSid: %s", sid[:12])
        # Enforce upper bound to prevent memory exhaustion from rapid webhook spam
        if len(validated_call_sids) > MAX_VALIDATED_SIDS:
            sorted_sids = sorted(validated_call_sids.items(), key=lambda x: x[1])
            excess = len(validated_call_sids) - MAX_VALIDATED_SIDS
            for sid, _ in sorted_sids[:excess]:
                validated_call_sids.pop(sid, None)
            log.warning("Pruned %d excess validated CallSids (limit: %d)", excess, MAX_VALIDATED_SIDS)

    def _check_rate_limit(client_ip: str) -> bool:
        """Return True if the request is within rate limits, False to reject."""
        now = time.monotonic()

        # Prune tracked IPs if over capacity (evict oldest)
        if client_ip not in _webhook_hits and len(_webhook_hits) >= WEBHOOK_RATE_MAX_IPS:
            oldest_ip = min(_webhook_hits, key=lambda ip: _webhook_hits[ip][-1] if _webhook_hits[ip] else 0)
            _webhook_hits.pop(oldest_ip, None)

        if client_ip not in _webhook_hits:
            _webhook_hits[client_ip] = collections.deque()

        hits = _webhook_hits[client_ip]

        # Slide the window: remove timestamps older than the window
        while hits and now - hits[0] > WEBHOOK_RATE_WINDOW_S:
            hits.popleft()

        if len(hits) >= WEBHOOK_RATE_LIMIT:
            return False

        hits.append(now)
        return True

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/incoming-call")
    async def incoming_call(request: Request):
        """SignalWire webhook for incoming calls."""
        # Rate limit by client IP
        client_ip = request.client.host if request.client else "unknown"
        if not _check_rate_limit(client_ip):
            log.warning("RATE LIMITED webhook from %s", client_ip)
            return Response(content="Too Many Requests", status_code=429)

        form = dict(await request.form())
        call_sid = form.get("CallSid", "")
        caller = form.get("From", "unknown")

        # Validate webhook signature
        if not _validate_webhook(config, request, form):
            log.warning("REJECTED invalid webhook signature from %s", client_ip)
            return Response(content="Forbidden", status_code=403)

        # Prune stale CallSids on each webhook
        _prune_stale_callsids()

        # Check concurrent call limit
        if len(active_calls) >= config.max_concurrent_calls:
            log.warning("REJECTED call — at capacity (%d/%d)",
                        len(active_calls), config.max_concurrent_calls)
            cxml = """<?xml version="1.0" encoding="UTF-8"?>
<Response><Say>I'm sorry, all lines are currently busy. Please try again later.</Say><Hangup/></Response>"""
            return Response(content=cxml, media_type="application/xml")

        log.info("── Incoming call from %s ──", caller)

        # Track this as a validated call (with timestamp for TTL)
        validated_call_sids[call_sid] = time.monotonic()

        # Build WebSocket URL
        host = config.public_host or request.headers.get("host", f"{config.host}:{config.port}")
        ws_url = f"wss://{host}/media-stream"

        # XML-escape all external values to prevent injection
        safe_ws_url = xml_escape(ws_url)
        safe_call_sid = xml_escape(call_sid)
        safe_caller = xml_escape(caller)

        cxml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{safe_ws_url}">
            <Parameter name="CallSid" value="{safe_call_sid}" />
            <Parameter name="CallerNumber" value="{safe_caller}" />
        </Stream>
    </Connect>
</Response>"""

        return Response(content=cxml, media_type="application/xml")

    @app.websocket("/media-stream")
    async def media_stream(ws: WebSocket):
        """SignalWire bidirectional media stream WebSocket."""

        await ws.accept()
        log.debug("WebSocket connected")

        handler: CallHandler | None = None
        stream_sid: str | None = None
        call_start_time: float = 0.0
        duration_task: asyncio.Task | None = None
        started = False  # Track whether we got a valid "start" event
        _finalized = False  # Reentrance guard for _finalize_call

        async def send_audio(mulaw_bytes: bytes):
            """Send mu-law audio back to SignalWire."""
            if stream_sid is None:
                return
            # Send in 2-second chunks (16000 bytes at 8kHz) to avoid
            # overwhelming SignalWire's buffer with hundreds of tiny messages.
            chunk_size = 16000
            for i in range(0, len(mulaw_bytes), chunk_size):
                chunk = mulaw_bytes[i:i + chunk_size]
                payload = base64.b64encode(chunk).decode("ascii")
                msg = {
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": payload},
                }
                try:
                    await ws.send_json(msg)
                except Exception:
                    return  # WebSocket closed, stop sending
                await asyncio.sleep(0)

        async def send_clear():
            """Send clear event to flush outbound audio on barge-in."""
            if stream_sid is None:
                return
            try:
                await ws.send_json({"event": "clear", "streamSid": stream_sid})
            except Exception:
                return  # WebSocket closed
            log.debug("Sent clear event")

        async def _enforce_max_duration():
            """Kill the call if it exceeds max duration."""
            try:
                await asyncio.sleep(config.max_call_duration_s)
            except asyncio.CancelledError:
                return
            log.warning("Call exceeded max duration (%ds), closing", config.max_call_duration_s)
            try:
                await ws.close(code=1000, reason="Max call duration exceeded")
            except Exception:
                pass

        async def _enforce_start_timeout():
            """Close connection if no valid 'start' event arrives in time."""
            try:
                await asyncio.sleep(WS_START_TIMEOUT_S)
            except asyncio.CancelledError:
                return
            if not started:
                log.warning("WebSocket timed out waiting for start event, closing")
                try:
                    await ws.close(code=4008, reason="Start timeout")
                except Exception:
                    pass

        # Start the connection timeout — auto-closes if no "start" arrives
        start_timeout_task = asyncio.create_task(_enforce_start_timeout())

        async def _finalize_call():
            """Run call teardown once: summary, persistence, notification, cleanup."""
            nonlocal handler, _finalized
            if _finalized or not handler:
                return
            _finalized = True

            h = handler
            handler = None  # Prevent concurrent access

            duration = max(0.0, time.perf_counter() - call_start_time) if call_start_time else 0.0
            transcript = list(h.transcript)
            summary = await h.on_stop()
            _log_call_end(h.caller_number, h.call_sid, duration, summary)
            _persist_call_metadata(
                config=config,
                caller_number=h.caller_number,
                call_sid=h.call_sid,
                duration=duration,
                summary=summary,
                transcript=transcript,
                recording_path=h.last_recording_path,
            )
            await _notify(config, h.caller_number, summary, duration, transcript)
            active_calls.pop(h.call_sid, None)

        async def _cancel_task(task: asyncio.Task | None):
            """Cancel an async task and await its completion."""
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        try:
            while True:
                raw = await ws.receive_text()

                # Drop oversized frames
                if len(raw) > WS_MAX_FRAME_SIZE:
                    log.warning("Dropping oversized WebSocket frame (%d bytes)", len(raw))
                    continue

                data = json.loads(raw)
                event = data.get("event")

                if event == "connected":
                    log.debug("Stream connected")

                elif event == "start":
                    start_obj = data.get("start", {})
                    stream_sid = data.get("streamSid") or start_obj.get("streamSid", "unknown")
                    custom_params = start_obj.get("customParameters", {})
                    call_sid = (custom_params.get("CallSid")
                                or start_obj.get("callSid")
                                or data.get("callSid", "unknown"))
                    caller_number = custom_params.get("CallerNumber", "unknown")

                    # Validate CallSid was from a signed webhook (with TTL check)
                    if call_sid not in validated_call_sids:
                        log.warning("REJECTED stream — CallSid %s not from validated webhook",
                                    call_sid[:16])
                        await ws.close(code=4003, reason="Invalid CallSid")
                        return

                    # Check TTL
                    sid_age = time.monotonic() - validated_call_sids.get(call_sid, 0)
                    if sid_age > CALLSID_TTL_S:
                        log.warning("REJECTED stream — CallSid %s expired (%.0fs old)",
                                    call_sid[:16], sid_age)
                        validated_call_sids.pop(call_sid, None)
                        await ws.close(code=4003, reason="CallSid expired")
                        return

                    validated_call_sids.pop(call_sid, None)
                    started = True

                    # Cancel the start timeout
                    await _cancel_task(start_timeout_task)

                    log.info("━━━ Call started  caller=%s  id=%s ━━━", caller_number, call_sid[:12])
                    call_start_time = time.perf_counter()

                    # Start max duration timer
                    duration_task = asyncio.create_task(_enforce_max_duration())

                    # Create per-call VAD
                    vad = SileroVAD(
                        model=vad_model,
                        speech_threshold=config.vad_speech_threshold,
                        silence_threshold_ms=config.vad_silence_threshold_ms,
                        min_speech_ms=config.vad_min_speech_ms,
                    )
                    handler = CallHandler(
                        config=config,
                        stt=stt,
                        tts=tts,
                        vad=vad,
                        call_sid=call_sid,
                        stream_sid=stream_sid,
                        caller_number=caller_number,
                        greeting_cache=greeting_cache,
                        silence_prompt_cache=silence_prompt_cache,
                    )
                    active_calls[call_sid] = handler
                    await handler.start(send_audio, send_clear)

                elif event == "media" and handler:
                    payload = data.get("media", {}).get("payload", "")
                    if payload:
                        # Reject oversized base64 payloads before decoding
                        if len(payload) > MAX_AUDIO_PAYLOAD_B64_LEN:
                            log.warning("Dropping oversized audio payload (%d bytes b64)", len(payload))
                            continue
                        try:
                            mulaw_bytes = base64.b64decode(payload, validate=True)
                        except binascii.Error:
                            continue  # Silently drop malformed frames
                        await handler.on_audio(mulaw_bytes)

                    # Capture local ref to avoid null-reference race with _finalize_call
                    h = handler
                    # Agent requested hangup (LLM emitted [HANGUP])
                    if h and h.hangup_requested:
                        log.debug("Agent requested call hangup")
                        await _finalize_call()
                        try:
                            await ws.close(code=1000, reason="Agent ended call")
                        except Exception:
                            pass
                        break

                elif event == "stop":
                    log.debug("Stream stopped")
                    await _finalize_call()
                    break

        except WebSocketDisconnect:
            log.debug("WebSocket disconnected")
            await _finalize_call()
        except Exception as e:
            log.error("WebSocket error: %s", type(e).__name__)
            await _finalize_call()
        finally:
            await _cancel_task(duration_task)
            await _cancel_task(start_timeout_task)

    # ── Demo mode endpoints ──

    MAX_DEMO_SESSIONS = 2
    _demo_session_count = 0

    @app.get("/demo/status")
    async def demo_status():
        """Runtime readiness checks used by the demo UI."""
        if not config.demo_mode:
            return Response(content="Demo mode not enabled. Start with: python main.py --demo",
                            status_code=403)

        checks: list[dict] = []

        def _add_check(check_id: str, ok: bool, message: str, severity: str = "error"):
            checks.append({
                "id": check_id,
                "ok": ok,
                "severity": severity,
                "message": message,
            })

        def _extract_model_ids(payload: dict) -> list[str]:
            model_ids: list[str] = []
            if isinstance(payload, dict):
                # OpenAI-compatible: {"data": [{"id": "..."}]}
                data = payload.get("data")
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            mid = str(item.get("id", "")).strip()
                            if mid:
                                model_ids.append(mid)
                # Ollama native tags endpoint: {"models": [{"name": "..."}]}
                tags = payload.get("models")
                if isinstance(tags, list):
                    for item in tags:
                        if isinstance(item, dict):
                            mid = str(item.get("name", "")).strip()
                            if mid:
                                model_ids.append(mid)
            return sorted(set(model_ids))

        llm_models: list[str] = []
        llm_ok = False
        llm_message = "Unknown LLM status"
        llm_base_url = (config.llm_base_url or "").rstrip("/")
        headers = {}
        if config.llm_api_key:
            headers["Authorization"] = f"Bearer {config.llm_api_key}"

        try:
            import httpx
            async with httpx.AsyncClient(timeout=2.5) as client:
                models_url = f"{llm_base_url}/models" if llm_base_url else ""
                if models_url:
                    resp = await client.get(models_url, headers=headers)
                    if resp.status_code == 200:
                        llm_ok = True
                        payload = resp.json()
                        llm_models = _extract_model_ids(payload)
                        llm_message = f"Connected to LLM endpoint ({models_url})"
                    else:
                        llm_message = (
                            f"LLM endpoint returned HTTP {resp.status_code} at {models_url}"
                        )

                # Fallback for Ollama native endpoint when /v1/models is unavailable.
                if not llm_ok and llm_base_url:
                    base = llm_base_url[:-3] if llm_base_url.endswith("/v1") else llm_base_url
                    tags_url = f"{base}/api/tags"
                    resp = await client.get(tags_url)
                    if resp.status_code == 200:
                        llm_ok = True
                        payload = resp.json()
                        llm_models = _extract_model_ids(payload)
                        llm_message = f"Connected to Ollama endpoint ({tags_url})"
        except Exception as e:
            llm_message = f"Could not reach LLM endpoint ({llm_base_url}): {type(e).__name__}"

        _add_check("llm_reachable", llm_ok, llm_message)

        if llm_ok:
            configured_model = (config.llm_model or "").strip()
            if configured_model:
                model_match = configured_model in llm_models if llm_models else True
                if model_match:
                    _add_check("llm_model", True, f"Configured model: {configured_model}", "info")
                else:
                    _add_check(
                        "llm_model",
                        False,
                        f"Configured model '{configured_model}' not found on endpoint.",
                        "warning",
                    )
            else:
                _add_check("llm_model", True, "No explicit model set; server default will be used.", "info")

        # TTS voice prompt status
        if config.tts_voice_prompt:
            voice_ok = Path(config.tts_voice_prompt).is_file()
            if voice_ok:
                _add_check("voice_prompt", True, f"Voice prompt found: {config.tts_voice_prompt}", "info")
            else:
                _add_check("voice_prompt", False, f"Voice prompt not found: {config.tts_voice_prompt}")
        else:
            _add_check("voice_prompt", True, "No custom voice prompt; using default Chatterbox voice.", "info")

        # Chatterbox model source status (prefer bundled local dir when available).
        bundle_dir: Path | None = None
        if config.tts_model_dir:
            bundle_dir = Path(config.tts_model_dir)
        else:
            default_bundle = Path(__file__).parent / "models" / "chatterbox"
            if default_bundle.is_dir():
                bundle_dir = default_bundle

        if bundle_dir:
            missing_files = [f for f in REQUIRED_MODEL_FILES if not (bundle_dir / f).is_file()]
            if missing_files:
                _add_check(
                    "tts_model_bundle",
                    False,
                    f"Bundled TTS model dir missing files ({', '.join(missing_files)}): {bundle_dir}",
                )
            else:
                _add_check("tts_model_bundle", True, f"Using bundled TTS model dir: {bundle_dir}", "info")
        else:
            _add_check(
                "tts_model_bundle",
                True,
                "No bundled TTS model dir detected; startup will use HF download/cache path.",
                "warning",
            )

        # Runtime informational checks
        _add_check("sessions", _demo_session_count < MAX_DEMO_SESSIONS,
                   f"Demo sessions in use: {_demo_session_count}/{MAX_DEMO_SESSIONS}",
                   "warning" if _demo_session_count >= MAX_DEMO_SESSIONS else "info")
        _add_check("capacity", len(active_calls) < config.max_concurrent_calls,
                   f"Call capacity in use: {len(active_calls)}/{config.max_concurrent_calls}",
                   "warning" if len(active_calls) >= config.max_concurrent_calls else "info")

        has_error = any((not c["ok"]) and c["severity"] == "error" for c in checks)
        has_warning = any((not c["ok"]) and c["severity"] == "warning" for c in checks)
        readiness = "ready" if not has_error and not has_warning else ("warning" if not has_error else "blocked")

        return {
            "ready": not has_error,
            "readiness": readiness,
            "checks": checks,
            "llm_provider": config.llm_provider,
            "llm_base_url": config.llm_base_url,
            "llm_model": config.llm_model,
            "demo_sessions": _demo_session_count,
            "max_demo_sessions": MAX_DEMO_SESSIONS,
            "active_calls": len(active_calls),
            "max_calls": config.max_concurrent_calls,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "available_models": llm_models[:50],
        }

    @app.get("/demo/config")
    async def demo_config():
        """Return editable configuration for the demo UI."""
        if not config.demo_mode:
            return Response(content="Demo mode not enabled. Start with: python main.py --demo",
                            status_code=403)

        values = {}
        for field in DEMO_SETTINGS_FIELDS:
            env_key = field["env"]
            attr = field["attr"]
            if attr and hasattr(config, attr):
                values[env_key] = _stringify_value(getattr(config, attr))
            else:
                values[env_key] = os.environ.get(env_key, "")

        return {
            "fields": DEMO_SETTINGS_FIELDS,
            "values": values,
        }

    @app.post("/demo/config")
    async def demo_config_update(request: Request):
        """Apply runtime config changes and/or persist them to .env."""
        if not config.demo_mode:
            return Response(content="Demo mode not enabled. Start with: python main.py --demo",
                            status_code=403)

        try:
            payload = await request.json()
        except Exception:
            return Response(content="Invalid JSON payload", status_code=400)

        values = payload.get("values", {})
        apply_runtime = bool(payload.get("apply_runtime", True))
        save = bool(payload.get("save", False))

        if not isinstance(values, dict):
            return Response(content="'values' must be an object", status_code=400)

        unknown = sorted([k for k in values if k not in _DEMO_FIELD_BY_ENV])
        if unknown:
            return Response(content=f"Unknown config keys: {', '.join(unknown)}", status_code=400)

        env_updates: dict[str, str] = {}
        attr_updates: dict[str, object] = {}
        applied_runtime: list[str] = []
        requires_restart: list[str] = []
        parse_errors: list[str] = []

        for env_key, raw in values.items():
            field = _DEMO_FIELD_BY_ENV[env_key]
            raw_str = "" if raw is None else str(raw)
            attr = field["attr"]
            if attr and hasattr(config, attr):
                current_raw = _stringify_value(getattr(config, attr))
            else:
                current_raw = os.environ.get(env_key, "")
            if raw_str == current_raw:
                continue

            env_updates[env_key] = raw_str

            try:
                # Empty string is valid for str fields, but not int/float.
                if field["type"] in ("int", "float") and raw_str.strip() == "":
                    raise ValueError("cannot be empty")
                typed = _typed_value(raw_str, field["type"])
            except Exception as e:
                parse_errors.append(f"{env_key}: {e}")
                continue

            if attr:
                attr_updates[attr] = typed

            if field["apply_runtime"]:
                applied_runtime.append(env_key)
            else:
                requires_restart.append(env_key)

        if parse_errors:
            return Response(
                content="Invalid config values:\n" + "\n".join(parse_errors),
                status_code=422,
            )

        if not env_updates:
            current_values = {}
            for field in DEMO_SETTINGS_FIELDS:
                env_key = field["env"]
                attr = field["attr"]
                if attr and hasattr(config, attr):
                    current_values[env_key] = _stringify_value(getattr(config, attr))
                else:
                    current_values[env_key] = os.environ.get(env_key, "")
            return {
                "ok": True,
                "applied_runtime": [],
                "requires_restart": [],
                "saved": False,
                "values": current_values,
            }

        # Validate resulting config values before applying/saving.
        try:
            candidate = copy.copy(config)
            for attr, typed in attr_updates.items():
                setattr(candidate, attr, typed)
            candidate.__post_init__()
        except Exception as e:
            return Response(content=str(e), status_code=422)

        if apply_runtime:
            for env_key in applied_runtime:
                field = _DEMO_FIELD_BY_ENV[env_key]
                attr = field["attr"]
                if attr and attr in attr_updates:
                    setattr(config, attr, attr_updates[attr])
                    stt_attr = _DEMO_STT_RUNTIME_ATTR_MAP.get(env_key)
                    if stt_attr and hasattr(stt, stt_attr):
                        setattr(stt, stt_attr, attr_updates[attr])
                os.environ[env_key] = env_updates[env_key]

        if save:
            env_path = Path(__file__).parent / ".env"
            _upsert_env_file(env_path, env_updates)

        current_values = {}
        for field in DEMO_SETTINGS_FIELDS:
            env_key = field["env"]
            attr = field["attr"]
            if attr and hasattr(config, attr):
                current_values[env_key] = _stringify_value(getattr(config, attr))
            else:
                current_values[env_key] = os.environ.get(env_key, "")

        return {
            "ok": True,
            "applied_runtime": applied_runtime if apply_runtime else [],
            "requires_restart": requires_restart,
            "saved": save,
            "values": current_values,
        }

    @app.post("/demo/restart")
    async def demo_restart():
        """Restart the HAL process (demo mode only)."""
        if not config.demo_mode:
            return Response(content="Demo mode not enabled. Start with: python main.py --demo",
                            status_code=403)
        if active_calls:
            return Response(content="Cannot restart while calls are active.", status_code=409)

        async def _restart_soon():
            await asyncio.sleep(0.6)
            try:
                python = sys.executable
                argv = [python] + sys.argv
                log.warning("Restarting HAL process via /demo/restart")
                os.execv(python, argv)
            except Exception as e:
                log.error("Failed to restart HAL process: %s", e)

        asyncio.create_task(_restart_soon())
        return {"ok": True, "message": "Restarting HAL..."}

    @app.websocket("/demo-stream")
    async def demo_stream(ws: WebSocket):
        """Browser-based demo: same pipeline as /media-stream, no SignalWire needed."""
        nonlocal _demo_session_count

        if not config.demo_mode:
            await ws.close(code=4003, reason="Demo mode not enabled")
            return

        if _demo_session_count >= MAX_DEMO_SESSIONS:
            await ws.close(code=4009, reason="Demo session limit reached")
            return

        # Check total capacity (shared with production calls)
        if len(active_calls) >= config.max_concurrent_calls:
            await ws.close(code=4009, reason="At capacity")
            return

        await ws.accept()
        _demo_session_count += 1

        call_sid = f"demo-{uuid.uuid4().hex[:12]}"
        stream_sid = f"demo-stream-{uuid.uuid4().hex[:8]}"
        caller_number = "demo-user"

        handler: CallHandler | None = None
        call_start_time: float = 0.0
        duration_task: asyncio.Task | None = None
        hangup_task: asyncio.Task | None = None
        _finalized = False

        async def send_audio(mulaw_bytes: bytes):
            """Send mu-law audio back to the browser."""
            chunk_size = 16000
            for i in range(0, len(mulaw_bytes), chunk_size):
                chunk = mulaw_bytes[i:i + chunk_size]
                payload = base64.b64encode(chunk).decode("ascii")
                msg = {
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": payload},
                }
                try:
                    await ws.send_json(msg)
                except Exception:
                    return
                await asyncio.sleep(0)

        async def send_clear():
            """Send clear event to flush outbound audio on barge-in."""
            try:
                await ws.send_json({"event": "clear", "streamSid": stream_sid})
            except Exception:
                return

        async def on_transcript(role: str, text: str):
            """Send transcript updates to the browser."""
            try:
                await ws.send_json({"event": "transcript", "role": role, "text": text})
            except Exception:
                pass

        async def _finalize_demo():
            nonlocal handler, _finalized
            if _finalized or not handler:
                return
            _finalized = True

            h = handler
            handler = None

            duration = max(0.0, time.perf_counter() - call_start_time) if call_start_time else 0.0
            transcript = list(h.transcript)
            summary = await h.on_stop()
            _log_call_end(h.caller_number, h.call_sid, duration, summary)
            _persist_call_metadata(
                config=config,
                caller_number=h.caller_number,
                call_sid=h.call_sid,
                duration=duration,
                summary=summary,
                transcript=transcript,
                recording_path=h.last_recording_path,
            )
            active_calls.pop(h.call_sid, None)

        async def _cancel_task(task: asyncio.Task | None):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        try:
            log.info("━━━ Demo started  id=%s ━━━", call_sid[:12])
            call_start_time = time.perf_counter()

            # Start max duration timer
            async def _enforce_max_duration_demo():
                try:
                    await asyncio.sleep(config.max_call_duration_s)
                except asyncio.CancelledError:
                    return
                log.warning("Demo call exceeded max duration (%ds), closing", config.max_call_duration_s)
                try:
                    await ws.close(code=1000, reason="Max call duration exceeded")
                except Exception:
                    pass

            duration_task = asyncio.create_task(_enforce_max_duration_demo())

            # Hangup polling task — checks hangup_requested independently of media frames.
            # In production, SignalWire sends a continuous audio stream so the hangup check
            # in the media handler fires reliably. In the browser demo, frames may stop
            # (e.g. after goodbye audio plays), so we need a separate poller.
            hangup_event = asyncio.Event()

            async def _poll_hangup():
                try:
                    while True:
                        await asyncio.sleep(0.2)
                        h = handler
                        if h and h.hangup_requested:
                            hangup_event.set()
                            return
                except asyncio.CancelledError:
                    return

            hangup_task = asyncio.create_task(_poll_hangup())

            # Create per-call VAD and handler
            vad = SileroVAD(
                model=vad_model,
                speech_threshold=config.vad_speech_threshold,
                silence_threshold_ms=config.vad_silence_threshold_ms,
                min_speech_ms=config.vad_min_speech_ms,
            )
            handler = CallHandler(
                config=config,
                stt=stt,
                tts=tts,
                vad=vad,
                call_sid=call_sid,
                stream_sid=stream_sid,
                caller_number=caller_number,
                greeting_cache=greeting_cache,
                silence_prompt_cache=silence_prompt_cache,
                on_transcript=on_transcript,
            )
            active_calls[call_sid] = handler
            await handler.start(send_audio, send_clear)

            # Main loop — same protocol as SignalWire, plus async hangup detection
            async def _recv_loop():
                """Process incoming WebSocket messages. Returns on stop or disconnect."""
                try:
                    while True:
                        raw = await ws.receive_text()
                        if len(raw) > WS_MAX_FRAME_SIZE:
                            continue

                        data = json.loads(raw)
                        event = data.get("event")

                        if event == "media" and handler:
                            payload = data.get("media", {}).get("payload", "")
                            if payload:
                                if len(payload) > MAX_AUDIO_PAYLOAD_B64_LEN:
                                    continue
                                try:
                                    mulaw_bytes = base64.b64decode(payload, validate=True)
                                except binascii.Error:
                                    continue
                                await handler.on_audio(mulaw_bytes)

                        elif event == "stop":
                            log.debug("Demo: client ended call")
                            return "stop"

                except WebSocketDisconnect:
                    log.debug("Demo WebSocket disconnected")
                    return "disconnect"

                return None

            # Run receive loop and hangup poller concurrently
            recv_task = asyncio.create_task(_recv_loop())
            done, pending = await asyncio.wait(
                [recv_task, hangup_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel whichever didn't finish
            for task in pending:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

            if hangup_event.is_set():
                log.debug("Demo: agent requested hangup")
                try:
                    await ws.send_json({"event": "hangup"})
                except Exception:
                    pass
                await _finalize_demo()
                try:
                    await ws.close(code=1000, reason="Agent ended call")
                except Exception:
                    pass
            else:
                # Client sent stop or WebSocket disconnected
                await _finalize_demo()

        except WebSocketDisconnect:
            log.debug("Demo WebSocket disconnected")
            await _finalize_demo()
        except Exception as e:
            log.error("Demo WebSocket error: %s", type(e).__name__)
            await _finalize_demo()
        finally:
            await _cancel_task(duration_task)
            await _cancel_task(hangup_task)
            _demo_session_count = max(0, _demo_session_count - 1)

    # ── Native HTML demo UI ──
    if config.demo_mode:
        from demo_ui import get_demo_html
        from fastapi.responses import HTMLResponse

        @app.get("/demo")
        async def demo_page():
            status_data = await demo_status()
            html = get_demo_html(status_data, config)
            return HTMLResponse(content=html)
        
        log.info("Native demo UI mounted at /demo")

    return app


def _log_call_end(caller_number: str, call_sid: str, duration: float, summary: str):
    """Print a clean call-end block to the terminal."""
    dur_str = _duration_human(duration)
    log.info("━━━ Call ended    caller=%s  id=%s  duration=%s ━━━", caller_number, call_sid[:12], dur_str)
    log.info("Summary: %s", summary)


async def _notify(config: Config, caller_number: str, summary: str,
                  duration: float, transcript: list[dict] | None = None):
    """Send call summary + transcript via ntfy."""
    if not config.ntfy_topic:
        return

    mins, secs = divmod(int(duration), 60)
    duration_str = f"{mins}m {secs}s" if mins else f"{secs}s"

    parts = [summary, ""]
    if transcript:
        parts.append("--- Transcript ---")
        for entry in transcript:
            role = "Caller" if entry["role"] == "caller" else "HAL"
            parts.append(f"{role}: {entry['text']}")
    body = "\n".join(parts)

    try:
        import httpx
        headers = {
            "Title": f"Call from {caller_number} ({duration_str})",
            "Priority": "high",
            "Tags": "phone",
        }
        # Add bearer token auth if configured
        if config.ntfy_token:
            headers["Authorization"] = f"Bearer {config.ntfy_token}"

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"https://ntfy.sh/{config.ntfy_topic}",
                content=body.encode("utf-8"),
                headers=headers,
                timeout=10,
            )
            log.debug("ntfy sent: %s", resp.status_code)
    except Exception as e:
        log.error("Failed to send ntfy: %s", e)
