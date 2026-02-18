"""FastAPI server with SignalWire webhook and WebSocket media stream endpoints."""

import asyncio
import base64
import binascii
import collections
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import Response

from config import Config
from audio import SileroVAD
from stt import SpeechToText
from tts import TTS
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
