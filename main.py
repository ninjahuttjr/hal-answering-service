"""
SuperCaller — Entry Point
===========================
Load environment, initialize models, start the server.
"""

import logging
import os
import time

import uvicorn
from dotenv import load_dotenv

from config import Config
from audio import load_silero_model
from stt import SpeechToText
from tts import TTS
from server import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)-20s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("supercaller")


def main():
    t_start = time.perf_counter()
    log.info("SuperCaller starting up...")

    # 1. Load .env
    load_dotenv()
    config = Config()
    log.info("Config loaded (LLM: %s, STT: %s)", config.llm_model, config.stt_model)

    # 2. Load STT model
    t0 = time.perf_counter()
    stt = SpeechToText(
        model_size=config.stt_model,
        device=config.stt_device,
        compute_type=config.stt_compute_type,
    )
    stt.load()
    log.info("STT loaded in %.1fs", time.perf_counter() - t0)

    # 3. Load TTS model (Chatterbox Turbo — kept hot on GPU)
    t0 = time.perf_counter()
    voice_prompt = config.tts_voice_prompt or None
    tts = TTS(voice_prompt=voice_prompt, device="cuda")
    log.info("TTS loaded in %.1fs", time.perf_counter() - t0)

    # 4. Load VAD model (Silero — shared across calls, cloned per-call)
    t0 = time.perf_counter()
    vad_model = load_silero_model()
    log.info("VAD loaded in %.1fs", time.perf_counter() - t0)

    # 5. Pre-record greetings (instant playback, no TTS delay per call)
    t0 = time.perf_counter()
    from prompts import get_greeting, SILENCE_PROMPTS
    greeting_cache = {}
    owner = config.owner_name
    for period in ("morning", "afternoon", "evening"):
        text = get_greeting(owner, time_of_day=period)
        chunks = []
        for chunk in tts.synthesize_mulaw_streaming(text):
            chunks.append(chunk["mulaw"])
        greeting_cache[period] = {"text": text, "mulaw": b"".join(chunks)}
        log.info("Pre-recorded greeting (%s): %d bytes, %.1fs audio",
                 period, len(greeting_cache[period]["mulaw"]),
                 len(greeting_cache[period]["mulaw"]) / 8000)
    log.info("Greetings pre-recorded in %.1fs", time.perf_counter() - t0)

    # 5b. Pre-record silence prompts (played when caller goes quiet)
    t0 = time.perf_counter()
    silence_prompt_cache = []
    for prompt_text in SILENCE_PROMPTS:
        chunks = []
        for chunk in tts.synthesize_mulaw_streaming(prompt_text):
            chunks.append(chunk["mulaw"])
        mulaw = b"".join(chunks)
        silence_prompt_cache.append({"text": prompt_text, "mulaw": mulaw})
        log.info("Pre-recorded silence prompt: %d bytes, %.1fs audio",
                 len(mulaw), len(mulaw) / 8000)
    log.info("Silence prompts pre-recorded in %.1fs", time.perf_counter() - t0)

    # 6. Ensure recordings directory exists
    os.makedirs(config.recordings_dir, exist_ok=True)
    log.info("Recordings dir: %s", os.path.abspath(config.recordings_dir))

    # 7. Create FastAPI app
    app = create_app(config, stt, tts, vad_model, greeting_cache, silence_prompt_cache)

    total = time.perf_counter() - t_start
    log.info("SuperCaller ready in %.1fs — listening on %s:%d", total, config.host, config.port)

    # 8. Run server (ws-max-size limits WebSocket frame size to 64KB)
    uvicorn.run(app, host=config.host, port=config.port, log_level="info",
                ws_max_size=65536)


if __name__ == "__main__":
    main()
