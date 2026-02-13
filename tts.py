"""
SuperCaller TTS Module — Chatterbox Turbo Integration
======================================================
Provides the ChatterboxTurboTTSWrapper for real-time phone call synthesis.
Model is loaded once at startup and kept hot on GPU. Conditionals (voice
embedding) are pre-baked so every generate() call is fast.

Output: 24kHz -> 8kHz resampled -> G.711 mu-law for telephony.
"""

import audioop
import logging
import time
import numpy as np
import torch

from chatterbox.tts_turbo import ChatterboxTurboTTS

log = logging.getLogger(__name__)

SAMPLE_RATE_CHATTERBOX = 24000
SAMPLE_RATE_8K = 8000


def _to_numpy(audio) -> np.ndarray:
    """Convert audio to numpy array (handles both tensors and ndarrays)."""
    if isinstance(audio, torch.Tensor):
        return audio.detach().cpu().numpy()
    return np.asarray(audio)


def _resample_linear(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio using linear interpolation."""
    if orig_sr == target_sr:
        return audio
    ratio = target_sr / orig_sr
    new_length = int(len(audio) * ratio)
    indices = np.arange(new_length) / ratio
    indices_floor = np.floor(indices).astype(np.int64)
    indices_ceil = np.minimum(indices_floor + 1, len(audio) - 1)
    frac = indices - indices_floor
    return (audio[indices_floor] * (1.0 - frac) + audio[indices_ceil] * frac).astype(np.float32)


def _pcm_to_mulaw(audio_float: np.ndarray) -> bytes:
    """Convert float audio [-1, 1] to G.711 mu-law bytes."""
    pcm16 = (np.clip(audio_float, -1.0, 1.0) * 32767).astype(np.int16)
    return audioop.lin2ulaw(pcm16.tobytes(), 2)


class TTS:
    """
    Chatterbox Turbo TTS wrapper for SuperCaller.

    The model and voice conditionals are loaded once at init and kept hot
    on GPU for the lifetime of this object. No cold-start penalty.

    Usage:
        tts = TTS()
        mulaw_bytes = tts.synthesize_mulaw("Hello, how can I help?")
    """

    def __init__(self, voice_prompt: str | None = None, device: str = "cuda"):
        """
        Args:
            voice_prompt: Path to a WAV file (>5s) for voice cloning.
                          If None, uses the built-in default voice.
            device: torch device ("cuda" or "cpu").
        """
        self.device = device
        self._voice_prompt = voice_prompt

        log.info("Loading Chatterbox Turbo model on %s...", device)
        t0 = time.perf_counter()
        self._model = ChatterboxTurboTTS.from_pretrained(device=device)
        load_ms = (time.perf_counter() - t0) * 1000
        log.info("Chatterbox Turbo loaded in %.0fms", load_ms)

        # Pre-bake voice conditionals if a custom voice prompt is provided
        if voice_prompt:
            log.info("Preparing voice conditionals from: %s", voice_prompt)
            self._model.prepare_conditionals(voice_prompt)

        # Warmup inference to eliminate first-call overhead
        log.info("TTS warmup inference...")
        t0 = time.perf_counter()
        self._model.generate("Warmup.")
        warmup_ms = (time.perf_counter() - t0) * 1000
        log.info("TTS warmup done in %.0fms", warmup_ms)

    def synthesize_mulaw(self, text: str) -> bytes:
        """Synthesize text and return 8kHz G.711 mu-law bytes for telephony."""
        if not text.strip():
            return b""

        wav_tensor = self._model.generate(text)
        audio_24k = _to_numpy(wav_tensor.squeeze())

        if len(audio_24k) == 0:
            return b""

        audio_8k = _resample_linear(audio_24k, SAMPLE_RATE_CHATTERBOX, SAMPLE_RATE_8K)
        return _pcm_to_mulaw(audio_8k)

    def synthesize_mulaw_streaming(self, text: str):
        """
        Yield mu-law bytes for the given text.

        Chatterbox Turbo doesn't have a native streaming API, so this
        generates the full audio and yields it as a single chunk. The
        call_handler already runs this in a thread, so it won't block
        the event loop.

        Yields:
            dict with "mulaw" (bytes) and "graphemes" (str).
        """
        mulaw = self.synthesize_mulaw(text)
        if mulaw:
            yield {"mulaw": mulaw, "graphemes": text}
