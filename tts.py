"""Chatterbox Turbo TTS wrapper — voice cloning and G.711 mu-law output for telephony."""

import audioop
import logging
import os
import time
import numpy as np
import torch

import chatterbox.tts_turbo as chatterbox_turbo
from chatterbox.tts_turbo import ChatterboxTurboTTS
from chatterbox.models.s3tokenizer import S3_SR
from chatterbox.models.s3gen import S3GEN_SR
from chatterbox.models.t3.modules.cond_enc import T3Cond

log = logging.getLogger(__name__)

# Workaround: chatterbox's prepare_conditionals promotes float32 -> float64 via
# Python float multiplication in norm_loudness, which crashes downstream LSTM layers.
# Try the original first; if it fails with a dtype error, fall back to our float32 version.
# Remove this patch once the upstream fix lands (track: github.com/resemble-ai/chatterbox).
_orig_prepare = ChatterboxTurboTTS.prepare_conditionals

# Workaround: on some environments (notably newer Python builds), Perth's implicit
# watermarker import path can resolve to None. Chatterbox then calls it unconditionally
# and crashes with "TypeError: 'NoneType' object is not callable".
# Fall back to DummyWatermarker so TTS can still run.
if getattr(chatterbox_turbo.perth, "PerthImplicitWatermarker", None) is None:
    if getattr(chatterbox_turbo.perth, "DummyWatermarker", None) is not None:
        log.warning("PerthImplicitWatermarker unavailable; using DummyWatermarker fallback")
        chatterbox_turbo.perth.PerthImplicitWatermarker = chatterbox_turbo.perth.DummyWatermarker

def _prepare_conditionals_f32(self, wav_fpath, exaggeration=0.5, norm_loudness=True):
    import librosa

    s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)
    s3gen_ref_wav = s3gen_ref_wav.astype(np.float32)

    assert len(s3gen_ref_wav) / _sr > 5.0, "Audio prompt must be longer than 5 seconds!"

    if norm_loudness:
        s3gen_ref_wav = self.norm_loudness(s3gen_ref_wav, _sr)
        s3gen_ref_wav = np.asarray(s3gen_ref_wav, dtype=np.float32)

    ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)
    ref_16k_wav = np.asarray(ref_16k_wav, dtype=np.float32)

    s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
    s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

    t3_cond_prompt_tokens = None
    if plen := self.t3.hp.speech_cond_prompt_len:
        s3_tokzr = self.s3gen.tokenizer
        t3_cond_prompt_tokens, _ = s3_tokzr.forward(
            [ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen
        )
        t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

    ve_embed = torch.from_numpy(
        self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR)
    )
    ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

    from chatterbox.tts_turbo import Conditionals
    t3_cond = T3Cond(
        speaker_emb=ve_embed,
        cond_prompt_speech_tokens=t3_cond_prompt_tokens,
        emotion_adv=exaggeration * torch.ones(1, 1, 1),
    ).to(device=self.device)
    self.conds = Conditionals(t3_cond, s3gen_ref_dict)

def _prepare_conditionals_safe(self, wav_fpath, exaggeration=0.5, norm_loudness=True):
    try:
        _orig_prepare(self, wav_fpath, exaggeration=exaggeration, norm_loudness=norm_loudness)
    except RuntimeError as e:
        if "expected scalar type Float" in str(e) or "dtype" in str(e).lower():
            log.warning("Upstream prepare_conditionals hit dtype bug, using float32 patch")
            _prepare_conditionals_f32(self, wav_fpath, exaggeration=exaggeration, norm_loudness=norm_loudness)
        else:
            raise

ChatterboxTurboTTS.prepare_conditionals = _prepare_conditionals_safe

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


MAX_TTS_TEXT_LENGTH = 2000  # Characters — prevent OOM on extremely long text


class TTS:
    """
    Chatterbox Turbo TTS wrapper for HAL Answering Service.

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
        import threading
        self.device = device
        self._voice_prompt = voice_prompt
        self._lock = threading.Lock()  # Protect model state from concurrent calls

        # Validate voice prompt file exists before loading model
        if voice_prompt and not os.path.isfile(voice_prompt):
            raise FileNotFoundError(f"TTS voice prompt file not found: {voice_prompt}")

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

        # Cap text length to prevent OOM
        if len(text) > MAX_TTS_TEXT_LENGTH:
            log.warning("TTS text truncated from %d to %d chars", len(text), MAX_TTS_TEXT_LENGTH)
            text = text[:MAX_TTS_TEXT_LENGTH]

        with self._lock:
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
