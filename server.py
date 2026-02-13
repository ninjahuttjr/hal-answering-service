"""
SuperCaller Server (Hardened)
==============================
FastAPI server with SignalWire webhook + WebSocket endpoints.
- Webhook signature validation via SignalWire RequestValidator
- WebSocket auth via validated CallSid tracking with TTL
- Max concurrent call limits
- Max call duration timeout
- XML-escaped CXML responses
- WebSocket connection timeout
"""

import asyncio
import base64
import binascii
import json
import logging
import time
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
            valid = validator.validate(str(request.url), form, signature)
        return valid
    except ImportError:
        log.error("signalwire.request_validator not available — REJECTING all webhooks. "
                  "Install the signalwire package.")
        return False
    except Exception as e:
        log.error("Webhook validation error: %s", e)
        return False


def create_app(config: Config, stt: SpeechToText, tts: TTS, vad_model,
               greeting_cache: dict | None = None,
               silence_prompt_cache: list | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="SuperCaller", docs_url=None, redoc_url=None, openapi_url=None)

    # Track active calls and validated call SIDs with timestamp for TTL
    active_calls: dict[str, CallHandler] = {}
    validated_call_sids: dict[str, float] = {}  # {call_sid: timestamp}

    def _prune_stale_callsids():
        """Remove validated CallSids older than TTL."""
        now = time.monotonic()
        stale = [sid for sid, ts in validated_call_sids.items()
                 if now - ts > CALLSID_TTL_S]
        for sid in stale:
            validated_call_sids.pop(sid, None)
            log.info("Pruned stale CallSid: %s", sid[:12])

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/incoming-call")
    async def incoming_call(request: Request):
        """SignalWire webhook for incoming calls."""
        form = dict(await request.form())
        call_sid = form.get("CallSid", "")
        caller = form.get("From", "unknown")
        called = form.get("To", "unknown")

        # Validate webhook signature
        if not _validate_webhook(config, request, form):
            log.warning("REJECTED invalid webhook signature from %s",
                        request.client.host if request.client else "unknown")
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

        log.info("Incoming call: %s -> %s", caller, called)

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
        log.info("WebSocket connected")

        handler: CallHandler | None = None
        stream_sid: str | None = None
        call_start_time: float = 0.0
        duration_task: asyncio.Task | None = None
        started = False  # Track whether we got a valid "start" event

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
                await ws.send_json(msg)
                await asyncio.sleep(0)

        async def send_clear():
            """Send clear event to flush outbound audio on barge-in."""
            if stream_sid is None:
                return
            await ws.send_json({"event": "clear", "streamSid": stream_sid})
            log.info("Sent clear event")

        async def _enforce_max_duration():
            """Kill the call if it exceeds max duration."""
            await asyncio.sleep(config.max_call_duration_s)
            log.warning("Call exceeded max duration (%ds), closing", config.max_call_duration_s)
            try:
                await ws.close(code=1000, reason="Max call duration exceeded")
            except Exception:
                pass

        async def _enforce_start_timeout():
            """Close connection if no valid 'start' event arrives in time."""
            await asyncio.sleep(WS_START_TIMEOUT_S)
            if not started:
                log.warning("WebSocket timed out waiting for start event, closing")
                try:
                    await ws.close(code=4008, reason="Start timeout")
                except Exception:
                    pass

        # Start the connection timeout — auto-closes if no "start" arrives
        start_timeout_task = asyncio.create_task(_enforce_start_timeout())

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
                    log.info("Stream connected")

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
                    if not start_timeout_task.done():
                        start_timeout_task.cancel()

                    log.info("Stream started: call=%s caller=%s", call_sid, caller_number)
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
                        try:
                            mulaw_bytes = base64.b64decode(payload, validate=True)
                        except binascii.Error:
                            continue  # Silently drop malformed frames
                        await handler.on_audio(mulaw_bytes)

                    # Agent requested hangup (LLM emitted [HANGUP])
                    if handler.hangup_requested:
                        log.info("Agent requested call hangup")
                        duration = time.perf_counter() - call_start_time
                        transcript = list(handler.transcript)
                        summary = await handler.on_stop()
                        await _notify(config, handler.caller_number, summary, duration, transcript)
                        active_calls.pop(handler.call_sid, None)
                        handler = None
                        try:
                            await ws.close(code=1000, reason="Agent ended call")
                        except Exception:
                            pass
                        break

                elif event == "stop":
                    log.info("Stream stopped")
                    if handler:
                        duration = time.perf_counter() - call_start_time
                        transcript = list(handler.transcript)
                        summary = await handler.on_stop()
                        await _notify(config, handler.caller_number, summary, duration, transcript)
                        active_calls.pop(handler.call_sid, None)
                        handler = None
                    break

        except WebSocketDisconnect:
            log.info("WebSocket disconnected")
            if handler:
                duration = time.perf_counter() - call_start_time
                transcript = list(handler.transcript)
                summary = await handler.on_stop()
                await _notify(config, handler.caller_number, summary, duration, transcript)
                active_calls.pop(handler.call_sid, None)
        except Exception as e:
            log.error("WebSocket error: %s", type(e).__name__)
            if handler:
                active_calls.pop(handler.call_sid, None)
        finally:
            if duration_task and not duration_task.done():
                duration_task.cancel()
            if not start_timeout_task.done():
                start_timeout_task.cancel()

    return app


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
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"https://ntfy.sh/{config.ntfy_topic}",
                content=body.encode("utf-8"),
                headers={
                    "Title": f"Call from {caller_number} ({duration_str})",
                    "Priority": "high",
                    "Tags": "phone",
                },
                timeout=10,
            )
            log.info("ntfy sent: %s", resp.status_code)
    except Exception as e:
        log.error("Failed to send ntfy: %s", e)
