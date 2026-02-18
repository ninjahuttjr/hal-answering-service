"""Per-call pipeline: VAD -> STT -> LLM -> TTS -> audio out."""

import asyncio
import audioop
import collections
import logging
import os
import re
import threading
import time
import wave
from datetime import datetime
from typing import Callable, Awaitable

import numpy as np

from config import Config
from audio import mulaw_decode, SileroVAD, SAMPLE_RATE_8K
from stt import SpeechToText
from llm import LLMClient
from tts import TTS
from prompts import get_greeting

log = logging.getLogger(__name__)

# Minimum audio duration (seconds) worth sending to STT.
MIN_AUDIO_DURATION_S = 0.8

# Cooldown after TTS finishes before listening (seconds).
ECHO_COOLDOWN_S = 0.5

# Tag the LLM emits to signal "hang up the call"
HANGUP_TAG = "[HANGUP]"

# --- Barge-in detection constants ---
BARGE_IN_RMS_DELTA = 0.03       # Must exceed ambient noise floor by this much
BARGE_IN_RMS_FLOOR = 0.01       # Absolute minimum threshold (even in dead silence)
BARGE_IN_FRAMES_REQUIRED = 6    # 6 × 20ms = 120ms window size for sliding window
BARGE_IN_FRAMES_THRESHOLD = 4   # 4 of 6 frames above threshold triggers barge-in
BARGE_IN_DELAY_S = 0.5          # Skip first 0.5s of bot speech (echo window)
RMS_EMA_ALPHA = 0.02            # Smoothing factor for noise floor tracking (post-calibration)
RMS_EMA_ALPHA_FAST = 0.1        # Fast alpha during initial calibration
BARGE_IN_CALIBRATION_FRAMES = 50  # ~1s of frames before barge-in activates

# --- Silence timeout constants ---
SILENCE_TIMEOUT_S = 20.0        # Ask "are you there?" after 20s silence
MAX_SILENCE_PROMPTS = 3         # After 3 with no response → hang up


class CallHandler:
    """Manages the full conversational loop for a single phone call."""

    def __init__(
        self,
        config: Config,
        stt: SpeechToText,
        tts: TTS,
        vad: SileroVAD,
        call_sid: str,
        stream_sid: str,
        caller_number: str,
        greeting_cache: dict | None = None,
        silence_prompt_cache: list | None = None,
    ):
        self.config = config
        self.stt = stt
        self.tts = tts
        self.call_sid = call_sid
        self.stream_sid = stream_sid
        self.caller_number = caller_number

        self.llm = LLMClient(config)
        self.vad = vad

        self.transcript: list[dict] = []
        self.speaking = False  # True while TTS audio is being sent
        self._send_audio: Callable[[bytes], Awaitable[None]] | None = None
        self._send_clear: Callable[[], Awaitable[None]] | None = None
        self._processing = False
        self._audio_frame_count = 0

        # Cancellation (used during hangup cleanup and barge-in)
        self._llm_cancel: asyncio.Event = asyncio.Event()

        # Pipeline task — decoupled from frame processing
        self._pipeline_task: asyncio.Task | None = None

        # Barge-in detection state
        self._barge_in_window: collections.deque = collections.deque(maxlen=BARGE_IN_FRAMES_REQUIRED)
        self._barge_in_audio_buffer: collections.deque = collections.deque(maxlen=25)  # ~500ms
        self._bot_speak_start_time: float = 0.0
        self._sentences_spoken: list[str] = []
        self._history_finalized = False
        self._rms_ema: float = 0.0  # Adaptive ambient noise floor
        self._ema_calibrated: bool = False  # True after enough frames to trust EMA

        # Lock protecting shared mutable state accessed from multiple async contexts
        # (_sentences_spoken, _rec_outbound, _history_finalized)
        self._state_lock = asyncio.Lock()

        # Recording state — wall-clock time for accurate alignment
        self._rec_inbound: list[bytes] = []  # Raw mu-law from caller (continuous, in order)
        self._rec_outbound: list[tuple[int, bytes]] = []  # (wall-clock sample offset, mu-law)
        self._rec_wall_start: float = 0.0  # perf_counter at stream start
        self.last_recording_path: str = ""

        # Playback cursor — models SignalWire's sequential audio buffer.
        self._playback_end_time: float = 0.0

        # Pre-recorded greetings (keyed by time-of-day: morning/afternoon/evening)
        self._greeting_cache = greeting_cache

        # Pre-recorded silence prompts
        self._silence_prompt_cache = silence_prompt_cache or []

        # Hangup state — set True when LLM emits [HANGUP], server checks this
        self.hangup_requested: bool = False

        # Silence timeout state
        self._silence_timer: asyncio.Task | None = None
        self._silence_prompt_index = 0
        self._silence_prompt_count = 0

    async def start(
        self,
        send_audio: Callable[[bytes], Awaitable[None]],
        send_clear: Callable[[], Awaitable[None]],
    ):
        """Called when the media stream starts. Sends the initial greeting."""
        self._send_audio = send_audio
        self._send_clear = send_clear
        self._rec_wall_start = time.perf_counter()

        # Brief settle time for SignalWire's media stream
        await asyncio.sleep(0.25)

        # Pick the right time-of-day greeting
        hour = datetime.now().hour
        if hour < 12:
            period = "morning"
        elif hour < 17:
            period = "afternoon"
        else:
            period = "evening"

        self.speaking = True

        if self._greeting_cache and period in self._greeting_cache:
            # Use pre-recorded greeting (instant — no TTS latency)
            cached = self._greeting_cache[period]
            greeting = cached["text"]
            mulaw_bytes = cached["mulaw"]

            # Track playback timing
            now = time.perf_counter()
            audio_duration = len(mulaw_bytes) / SAMPLE_RATE_8K
            self._playback_end_time = now + audio_duration

            # Record outbound audio
            rec_offset = int((now - self._rec_wall_start) * SAMPLE_RATE_8K)
            self._rec_outbound.append((rec_offset, mulaw_bytes))

            await self._send_audio(mulaw_bytes)
            log.debug("[%s] Greeting sent (%d bytes, %.1fs audio)",
                      self.call_sid, len(mulaw_bytes), audio_duration)
        else:
            # Fallback: generate on the fly
            greeting = get_greeting(self.config.owner_name)
            log.debug("[%s] Sending greeting (live TTS): %s", self.call_sid, greeting)
            await self._synthesize_and_send(greeting)

        self.transcript.append({"role": "agent", "text": greeting})
        self.llm.history.append({"role": "assistant", "content": greeting})
        self.speaking = False

        # Start silence timer
        self._reset_silence_timer()

    def _in_echo_cooldown(self) -> bool:
        if self._playback_end_time == 0.0:
            return False
        return time.perf_counter() < (self._playback_end_time + ECHO_COOLDOWN_S)

    async def on_audio(self, mulaw_data: bytes):
        """Called for each inbound audio frame from SignalWire.

        Three states:
          STATE 1 (speaking): Buffer audio, detect barge-in via RMS
          STATE 2 (echo cooldown): Ignore frames
          STATE 3 (normal listening): Feed VAD, update noise floor EMA
        """
        # Record inbound audio (continuous stream, concatenated in order)
        self._rec_inbound.append(mulaw_data)

        pcm = mulaw_decode(mulaw_data)
        self._audio_frame_count += 1

        # Compute RMS for all states
        rms = np.sqrt(np.mean(pcm ** 2)) if len(pcm) > 0 else 0.0

        # EMA calibration: use fast alpha initially, slow alpha once calibrated
        # Updated in STATE 2 and STATE 3 (not STATE 1 — can't distinguish speech from noise)
        if not self._ema_calibrated and self._audio_frame_count >= BARGE_IN_CALIBRATION_FRAMES:
            self._ema_calibrated = True
            log.debug("[%s] Noise floor calibrated: EMA=%.4f threshold=%.4f",
                      self.call_sid, self._rms_ema,
                      max(BARGE_IN_RMS_FLOOR, self._rms_ema + BARGE_IN_RMS_DELTA))

        # Debug logging every ~2 seconds
        if self._audio_frame_count % 100 == 1:
            log.debug("[%s] frame #%d: RMS=%.4f speaking=%s vad=%s cooldown=%s ema=%.4f cal=%s",
                      self.call_sid, self._audio_frame_count, rms,
                      self.speaking, self.vad.speaking, self._in_echo_cooldown(),
                      self._rms_ema, self._ema_calibrated)

        # === STATE 1: SPEAKING — detect barge-in ===
        if self.speaking:
            # Buffer PCM in circular deque (captures caller's first words)
            self._barge_in_audio_buffer.append(pcm)

            # No barge-in until noise floor is calibrated
            if not self._ema_calibrated:
                return

            # Skip echo window at start of bot speech
            if self._bot_speak_start_time > 0.0:
                elapsed = time.perf_counter() - self._bot_speak_start_time
                if elapsed < BARGE_IN_DELAY_S:
                    return

            # Adaptive barge-in threshold — sliding window (N of M frames)
            barge_threshold = max(BARGE_IN_RMS_FLOOR, self._rms_ema + BARGE_IN_RMS_DELTA)
            self._barge_in_window.append(1 if rms > barge_threshold else 0)

            if (len(self._barge_in_window) >= BARGE_IN_FRAMES_REQUIRED
                    and sum(self._barge_in_window) >= BARGE_IN_FRAMES_THRESHOLD):
                log.info("[%s] Barge-in detected! RMS=%.4f threshold=%.4f hits=%d/%d",
                         self.call_sid, rms, barge_threshold,
                         sum(self._barge_in_window), len(self._barge_in_window))
                await self._handle_barge_in()
            return

        # === STATE 2: ECHO COOLDOWN — bot audio still playing from buffer ===
        if self._in_echo_cooldown():
            # Update EMA during cooldown for calibration (especially during greeting)
            ema_alpha = RMS_EMA_ALPHA if self._ema_calibrated else RMS_EMA_ALPHA_FAST
            self._rms_ema = ema_alpha * rms + (1 - ema_alpha) * self._rms_ema

            # No barge-in until noise floor is calibrated
            if not self._ema_calibrated:
                return

            # Run barge-in detection — pipeline finished sending TTS bytes
            # quickly, but SignalWire is still playing them back.
            self._barge_in_audio_buffer.append(pcm)
            barge_threshold = max(BARGE_IN_RMS_FLOOR, self._rms_ema + BARGE_IN_RMS_DELTA)
            self._barge_in_window.append(1 if rms > barge_threshold else 0)

            if (len(self._barge_in_window) >= BARGE_IN_FRAMES_REQUIRED
                    and sum(self._barge_in_window) >= BARGE_IN_FRAMES_THRESHOLD):
                log.info("[%s] Barge-in during cooldown! RMS=%.4f threshold=%.4f",
                         self.call_sid, rms, barge_threshold)
                await self._handle_barge_in()
            return

        # === STATE 3: NORMAL LISTENING ===
        result = self.vad.feed(pcm)

        # Update adaptive noise floor during confirmed silence
        # (VAD not tracking speech, bot not talking)
        if not self.vad.speaking:
            ema_alpha = RMS_EMA_ALPHA if self._ema_calibrated else RMS_EMA_ALPHA_FAST
            self._rms_ema = ema_alpha * rms + (1 - ema_alpha) * self._rms_ema

        if result == "end_of_speech" and not self._processing:
            self._processing = True
            self._pipeline_task = asyncio.create_task(self._process_utterance_wrapper())

    async def _process_utterance_wrapper(self):
        """Thin wrapper around _process_utterance with cleanup in finally."""
        try:
            await self._process_utterance()
        except Exception as e:
            log.error("[%s] Pipeline error: %s: %s", self.call_sid, type(e).__name__, e)
        finally:
            self._processing = False
            self.vad.reset()

    async def _process_utterance(self):
        """Run the full STT -> LLM -> TTS pipeline for one caller utterance."""
        audio_8k = self.vad.get_audio()

        if len(audio_8k) == 0:
            return

        duration_s = len(audio_8k) / SAMPLE_RATE_8K
        if duration_s < MIN_AUDIO_DURATION_S:
            log.info("[%s] Audio too short (%.2fs), skipping", self.call_sid, duration_s)
            return

        # STT
        t0 = time.perf_counter()
        try:
            text = await asyncio.to_thread(self.stt.transcribe, audio_8k, SAMPLE_RATE_8K)
        except Exception as e:
            log.error("[%s] STT transcription failed: %s", self.call_sid, e)
            return
        stt_ms = (time.perf_counter() - t0) * 1000
        log.info('[%s] Caller: "%s"  (%.0fms)', self.call_sid, text, stt_ms)

        if not text.strip():
            log.debug("[%s] STT returned empty text for %.2fs audio, skipping", self.call_sid, duration_s)
            return

        # Cap transcribed text length to limit prompt injection surface
        if len(text) > 500:
            text = text[:500]
            log.warning("[%s] STT text truncated to 500 chars", self.call_sid)

        self.transcript.append({"role": "caller", "text": text})

        # Add user message to LLM history (caller is responsible now)
        self.llm.add_user_message(text)

        # Reset sentence tracking for this pipeline run (under lock)
        async with self._state_lock:
            self._sentences_spoken = []
            self._history_finalized = False

        # LLM -> TTS with true pipelining
        self.speaking = True
        self._bot_speak_start_time = time.perf_counter()
        self._barge_in_window.clear()
        self._barge_in_audio_buffer.clear()
        self._llm_cancel.clear()

        # Shared stop event for the LLM thread (checked before each queue put)
        llm_stop = threading.Event()

        t1 = time.perf_counter()
        first_sentence = True
        hangup_after = False

        # Bridge sync LLM generator to async queue in real-time
        sentence_queue: asyncio.Queue[tuple[str, bool] | None] = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _llm_thread():
            """Runs in thread — pushes sentences to queue as they arrive."""
            try:
                for sentence, is_final in self.llm.chat_stream_sentences(text):
                    if self._llm_cancel.is_set() or llm_stop.is_set():
                        break
                    try:
                        asyncio.run_coroutine_threadsafe(
                            sentence_queue.put((sentence, is_final)), loop
                        ).result(timeout=5)
                    except Exception:
                        break  # Loop/queue gone, exit cleanly
            except Exception as e:
                log.error("[%s] LLM error: %s", self.call_sid, e)
            finally:
                if not llm_stop.is_set():
                    try:
                        asyncio.run_coroutine_threadsafe(
                            sentence_queue.put(None), loop
                        ).result(timeout=5)
                    except Exception:
                        pass  # Loop/queue gone

        llm_thread_future = loop.run_in_executor(None, _llm_thread)

        try:
            while True:
                # Check for cancellation while waiting
                try:
                    item = await asyncio.wait_for(sentence_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    if self._llm_cancel.is_set():
                        break
                    continue

                if item is None:
                    break

                sentence, is_final = item

                if self._llm_cancel.is_set():
                    log.debug("[%s] Interrupted, stopping pipeline", self.call_sid)
                    break

                # Detect [HANGUP] tag from LLM
                if HANGUP_TAG in sentence.upper():
                    sentence = re.sub(r'\[HANGUP\]', '', sentence, flags=re.IGNORECASE).strip()
                    # Strip leftover punctuation-only residue (e.g. lone ".")
                    if sentence and not any(c.isalnum() for c in sentence):
                        sentence = ""
                    hangup_after = True

                if first_sentence:
                    llm_ms = (time.perf_counter() - t1) * 1000
                    log.info('[%s] HAL: "%s"  (%.0fms)', self.call_sid, sentence, llm_ms)
                    first_sentence = False

                if sentence:
                    # Filter out [interrupted] markers leaked from history
                    if sentence.strip().strip('.!?') == '[interrupted]':
                        log.warning("[%s] LLM echoed [interrupted], skipping", self.call_sid)
                        continue

                    await self._synthesize_and_send(sentence)

                    if self._llm_cancel.is_set():
                        break

                    async with self._state_lock:
                        self._sentences_spoken.append(sentence)
                    self.transcript.append({"role": "agent", "text": sentence})

                if hangup_after:
                    log.info("[%s] LLM requested hangup", self.call_sid)
                    break

        finally:
            self._llm_cancel.set()  # Ensure thread exits
            llm_stop.set()  # Signal thread to stop before queue puts
            try:
                await asyncio.wait_for(llm_thread_future, timeout=3)
            except (asyncio.TimeoutError, Exception):
                log.warning("[%s] LLM thread did not exit cleanly", self.call_sid)

            # Finalize history (unless barge-in already did it)
            async with self._state_lock:
                if not self._history_finalized:
                    self._finalize_history()

        self.speaking = False
        self._bot_speak_start_time = 0.0

        if hangup_after:
            # Wait for SignalWire to finish playing ALL queued audio
            remaining = max(0, self._playback_end_time - time.perf_counter()) + 0.5
            log.debug("[%s] Waiting %.1fs for goodbye audio to play", self.call_sid, remaining)
            await asyncio.sleep(remaining)
            self.hangup_requested = True
            log.debug("[%s] Hangup flag set", self.call_sid)
        else:
            # Reset silence timer after normal pipeline completion
            self._reset_silence_timer()

    def _finalize_history(self):
        """Store spoken sentences in LLM history as the assistant message.

        MUST be called while holding self._state_lock.
        """
        if self._history_finalized:
            return
        self._history_finalized = True

        # Snapshot under lock
        spoken = list(self._sentences_spoken)

        if spoken:
            clean_response = ' '.join(spoken)
            self.llm.history.append({"role": "assistant", "content": clean_response})
            log.debug("[%s] History finalized: %s", self.call_sid, clean_response[:100])
        else:
            # Nothing spoken — add fallback to maintain turn alternation
            self.llm.history.append({"role": "assistant", "content": "Could you say that again?"})
            log.warning("[%s] No sentences spoken, added fallback to history", self.call_sid)

    def _finalize_interrupted_history(self):
        """Store spoken sentences + [interrupted] marker after barge-in.

        MUST be called while holding self._state_lock.
        """
        if self._history_finalized:
            return
        self._history_finalized = True

        # Guard: only add assistant message if there's a user message to respond to.
        # Barge-in during greeting/silence cooldown has no user message — don't pollute history.
        if not self.llm.history or self.llm.history[-1]["role"] != "user":
            log.debug("[%s] Barge-in with no pending user message, skipping history", self.call_sid)
            return

        # Snapshot under lock
        spoken = list(self._sentences_spoken)

        if spoken:
            clean_response = ' '.join(spoken) + " [interrupted]"
            self.llm.history.append({"role": "assistant", "content": clean_response})
            log.debug("[%s] Interrupted history finalized: %s", self.call_sid, clean_response[:100])
        else:
            # Bot was interrupted before speaking any sentence — remove the orphan user message
            # since the bot never responded, it's as if the exchange never happened
            removed = self.llm.history.pop()
            log.debug("[%s] Interrupted before speech, removed orphan user msg: %s",
                      self.call_sid, removed["content"][:60])

    async def _handle_barge_in(self):
        """Handle caller barge-in: stop pipeline, flush audio, seed VAD."""
        log.debug("[%s] Handling barge-in", self.call_sid)

        # 1. Signal pipeline to stop
        self._llm_cancel.set()

        # 2. Store spoken sentences + [interrupted] in LLM history (under lock)
        async with self._state_lock:
            self._finalize_interrupted_history()

        # 3. Flush SignalWire's queued outbound audio
        if self._send_clear:
            await self._send_clear()

        # 4. Trim outbound recording — remove audio that was flushed (never heard)
        barge_sample = int((time.perf_counter() - self._rec_wall_start) * SAMPLE_RATE_8K)
        async with self._state_lock:
            trimmed = []
            for offset, data in self._rec_outbound:
                end = offset + len(data)
                if offset >= barge_sample:
                    continue  # Queued but never played — discard entirely
                if end > barge_sample:
                    # Partially played — keep only what was heard
                    trimmed.append((offset, data[:barge_sample - offset]))
                else:
                    trimmed.append((offset, data))
            self._rec_outbound = trimmed

        # 5. Reset playback model
        self._playback_end_time = 0.0

        # 6. Transition to listening
        self.speaking = False
        self._bot_speak_start_time = 0.0
        self._barge_in_window.clear()

        # 7. Wait for pipeline task to finish FIRST — its finally block calls
        #    vad.reset(), so we must let it run before we seed the VAD.
        if self._pipeline_task and not self._pipeline_task.done():
            try:
                await asyncio.wait_for(self._pipeline_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                log.warning("[%s] Pipeline task did not finish in time after barge-in",
                            self.call_sid)

        # 8. NOW seed VAD with barge-in speech (after pipeline's finally has run)
        try:
            self.vad.reset()
            for buffered_pcm in self._barge_in_audio_buffer:
                self.vad.feed(buffered_pcm)
        except Exception as e:
            log.error("[%s] VAD seeding failed after barge-in: %s", self.call_sid, e)
            self.vad.reset()
        finally:
            self._barge_in_audio_buffer.clear()

        # 9. Allow new pipelines
        self._processing = False

        # Reset silence timer
        self._reset_silence_timer()

    # --- Silence timeout ---

    def _reset_silence_timer(self):
        """Cancel any existing silence timer and start a new one."""
        if self._silence_timer and not self._silence_timer.done():
            self._silence_timer.cancel()
        self._silence_timer = asyncio.create_task(self._silence_timeout_loop())

    async def _silence_timeout_loop(self):
        """Fire after SILENCE_TIMEOUT_S of silence. Play a prompt, restart timer."""
        try:
            await asyncio.sleep(SILENCE_TIMEOUT_S)
        except asyncio.CancelledError:
            return

        # Don't interrupt if bot is speaking, processing, or caller is mid-speech
        if self.speaking or self._processing or self.vad.speaking:
            # Retry — restart the timer (properly cancel old task first)
            self._reset_silence_timer()
            return

        if not self._silence_prompt_cache:
            log.warning("[%s] No silence prompts cached, restarting timer anyway", self.call_sid)
            self._reset_silence_timer()
            return

        self._silence_prompt_count += 1
        log.info("[%s] Silence timeout #%d", self.call_sid, self._silence_prompt_count)

        if self._silence_prompt_count > MAX_SILENCE_PROMPTS:
            log.info("[%s] Max silence prompts reached, hanging up", self.call_sid)
            self.hangup_requested = True
            return

        # Pick next prompt (cycling through list)
        cached = self._silence_prompt_cache[self._silence_prompt_index % len(self._silence_prompt_cache)]
        self._silence_prompt_index += 1

        prompt_text = cached["text"]
        mulaw_bytes = cached["mulaw"]

        # Play it
        self.speaking = True
        self._bot_speak_start_time = time.perf_counter()
        self._barge_in_count = 0
        self._barge_in_audio_buffer.clear()

        if self._send_audio and mulaw_bytes:
            now = time.perf_counter()
            audio_duration = len(mulaw_bytes) / SAMPLE_RATE_8K
            self._playback_end_time = now + audio_duration

            # Record outbound audio (under lock)
            rec_offset = int((now - self._rec_wall_start) * SAMPLE_RATE_8K)
            async with self._state_lock:
                self._rec_outbound.append((rec_offset, mulaw_bytes))

            try:
                await self._send_audio(mulaw_bytes)
            except Exception:
                log.debug("[%s] Silence prompt send failed (WebSocket closed)", self.call_sid)
                self.speaking = False
                self._bot_speak_start_time = 0.0
                return
            log.info("[%s] Silence prompt sent: %s (%.1fs audio)",
                     self.call_sid, prompt_text, audio_duration)

        # Add to transcript and LLM history (bot "remembers" it asked)
        self.transcript.append({"role": "agent", "text": prompt_text})
        self.llm.history.append({"role": "assistant", "content": prompt_text})

        self.speaking = False
        self._bot_speak_start_time = 0.0

        # Restart timer for next silence check
        self._reset_silence_timer()

    async def _synthesize_and_send(self, text: str):
        """Synthesize text to mu-law audio and send to SignalWire."""
        if not self._send_audio or self._llm_cancel.is_set():
            return

        t0 = time.perf_counter()

        def _generate():
            chunks = []
            try:
                for chunk in self.tts.synthesize_mulaw_streaming(text):
                    if self._llm_cancel.is_set():
                        return b""
                    chunks.append(chunk["mulaw"])
            except Exception as e:
                log.error("[%s] TTS synthesis failed: %s", self.call_sid, e)
                return b""
            return b"".join(chunks)

        mulaw_bytes = await asyncio.to_thread(_generate)
        tts_ms = (time.perf_counter() - t0) * 1000

        if mulaw_bytes and not self._llm_cancel.is_set():
            # Model SignalWire's sequential playback buffer:
            # Audio doesn't play at send time — it queues after whatever's already playing.
            now = time.perf_counter()
            playback_start = max(now, self._playback_end_time)
            audio_duration = len(mulaw_bytes) / SAMPLE_RATE_8K
            self._playback_end_time = playback_start + audio_duration

            # Record outbound audio at estimated PLAYBACK position (not send time)
            rec_offset = int((playback_start - self._rec_wall_start) * SAMPLE_RATE_8K)
            async with self._state_lock:
                self._rec_outbound.append((rec_offset, mulaw_bytes))

            try:
                await self._send_audio(mulaw_bytes)
            except Exception:
                log.debug("[%s] TTS send failed (WebSocket closed)", self.call_sid)
                return
            log.debug("[%s] TTS (%.0fms, %d bytes, %.1fs audio): %s",
                      self.call_sid, tts_ms, len(mulaw_bytes),
                      audio_duration, text[:60])

    def _save_recording(self):
        """Save call as mono WAV — both parties mixed, like a real phone recording."""
        if not self._rec_inbound:
            log.info("[%s] No audio to record", self.call_sid)
            self.last_recording_path = ""
            return

        try:
            # CALLER: continuous mu-law stream, decode to int32 for mixing headroom
            inbound_mulaw = b"".join(self._rec_inbound)
            inbound_pcm = np.frombuffer(
                audioop.ulaw2lin(inbound_mulaw, 2), dtype=np.int16
            ).astype(np.int32)

            # Total duration — use wall clock, ensure all outbound fits
            wall_duration = time.perf_counter() - self._rec_wall_start
            total_samples = max(len(inbound_pcm), int(wall_duration * SAMPLE_RATE_8K))
            for offset, data in self._rec_outbound:
                total_samples = max(total_samples, offset + len(data))

            # Create mix buffer (int32 to prevent clipping during addition)
            mixed = np.zeros(total_samples, dtype=np.int32)

            # Add caller audio (starts at sample 0, continuous)
            mixed[:len(inbound_pcm)] += inbound_pcm

            # Add agent audio at estimated playback positions
            for sample_offset, mulaw_data in self._rec_outbound:
                pcm = np.frombuffer(
                    audioop.ulaw2lin(mulaw_data, 2), dtype=np.int16
                ).astype(np.int32)
                end = min(sample_offset + len(pcm), total_samples)
                n = end - sample_offset
                if n > 0 and sample_offset >= 0:
                    mixed[sample_offset:end] += pcm[:n]

            # Clip to int16 range and convert
            mixed = np.clip(mixed, -32768, 32767).astype(np.int16)

            # Save mono WAV
            rec_dir = self.config.recordings_dir
            os.makedirs(rec_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # Microsecond precision
            # Strip everything except alphanumeric chars to prevent path traversal
            safe_number = re.sub(r'[^a-zA-Z0-9]', '', self.caller_number)
            safe_sid = re.sub(r'[^a-zA-Z0-9]', '', self.call_sid[:8])
            filename = f"{timestamp}_{safe_number}_{safe_sid}.wav"
            filepath = os.path.join(rec_dir, filename)
            # Final safety check: ensure path is within recordings dir
            if not os.path.realpath(filepath).startswith(os.path.realpath(rec_dir)):
                log.error("[%s] Path traversal blocked: %s", self.call_sid, filepath)
                return

            with wave.open(filepath, "wb") as wf:
                wf.setnchannels(1)  # Mono — both parties mixed
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE_8K)
                wf.writeframes(mixed.tobytes())

            self.last_recording_path = filepath
            duration_s = total_samples / SAMPLE_RATE_8K
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            log.info("[%s] Recording: %s (%.1fs, %.1fMB)",
                     self.call_sid, filepath, duration_s, size_mb)

        except Exception as e:
            self.last_recording_path = ""
            log.error("[%s] Failed to save recording: %s", self.call_sid, e, exc_info=True)

    async def on_stop(self) -> str:
        """Called when the call ends. Generates a summary."""
        log.debug("[%s] Generating summary...", self.call_sid)
        self._llm_cancel.set()  # Kill any in-progress generation

        # Cancel silence timer
        if self._silence_timer and not self._silence_timer.done():
            self._silence_timer.cancel()

        # Wait for pipeline task to finish
        if self._pipeline_task and not self._pipeline_task.done():
            try:
                await asyncio.wait_for(self._pipeline_task, timeout=3.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        # Save recording in background thread (I/O)
        await asyncio.to_thread(self._save_recording)

        # Only summarize if the caller actually said something.
        # Transcript always contains the bot greeting, so check for caller entries.
        caller_messages = [t for t in self.transcript if t["role"] == "caller"]
        if not caller_messages:
            return f"Call from {self.caller_number}: Caller hung up without speaking."

        try:
            summary = await asyncio.to_thread(self.llm.get_summary, self.transcript)
            log.debug("[%s] Summary: %s", self.call_sid, summary)
            return summary
        except Exception as e:
            log.error("[%s] Summary generation failed: %s", self.call_sid, e)
            lines = [f"{t['role']}: {t['text']}" for t in self.transcript[:5]]
            return f"Call from {self.caller_number}:\n" + "\n".join(lines)
