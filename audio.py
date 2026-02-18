"""G.711 mu-law codec, resampling, and Silero VAD for telephony audio."""

import audioop
import logging
import numpy as np
import torch

log = logging.getLogger(__name__)

SAMPLE_RATE_8K = 8000
SILERO_CHUNK = 256  # 256 samples at 8kHz = 32ms per chunk


def mulaw_decode(data: bytes) -> np.ndarray:
    """Decode G.711 mu-law bytes to float32 audio in [-1, 1]."""
    pcm_bytes = audioop.ulaw2lin(data, 2)
    pcm16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    return pcm16.astype(np.float32) / 32768.0


def mulaw_encode(audio: np.ndarray) -> bytes:
    """Encode float32 audio [-1, 1] to G.711 mu-law bytes."""
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    return audioop.lin2ulaw(pcm16.tobytes(), 2)


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
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


def load_silero_model():
    """Load the Silero VAD model once (v6+). Returns the JIT model."""
    model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
    )
    model.eval()
    log.info("Silero VAD model loaded")
    return model


class SileroVAD:
    """
    Silero-based streaming Voice Activity Detector for telephony audio.

    Uses the Silero VAD model (same one used by LiveKit, Deepgram, etc.)
    for robust speech detection in noisy environments. Buffers 8kHz frames
    to 256-sample chunks as required by Silero.
    """

    def __init__(self, model, speech_threshold: float = 0.5,
                 silence_threshold_ms: int = 400,
                 min_speech_ms: int = 250):
        """
        Args:
            model: Pre-loaded Silero VAD JIT model (from load_silero_model).
            speech_threshold: Silero probability above this = speech (0.0-1.0).
            silence_threshold_ms: Silence duration to trigger end-of-speech.
            min_speech_ms: Minimum speech duration before we consider it real.
        """
        self.speech_threshold = speech_threshold
        # Convert ms to Silero chunk count (each chunk = 32ms at 8kHz/256 samples)
        self._silence_chunks = max(1, silence_threshold_ms // 32)
        self._min_speech_chunks = max(1, min_speech_ms // 32)

        # Use pre-loaded model (cloned for per-call isolation)
        import copy
        self._model = copy.deepcopy(model)
        self._model.reset_states()
        log.info("SileroVAD instance created (threshold=%.2f, silence=%dms, min_speech=%dms)",
                 speech_threshold, silence_threshold_ms, min_speech_ms)

        # State
        self._pending = np.array([], dtype=np.float32)  # Buffer for incomplete chunks
        self._buffer: list[np.ndarray] = []  # Accumulated speech audio
        self._speaking = False
        self._silent_count = 0
        self._speech_count = 0

        # Pre-buffer: keep last N chunks of silence so we capture speech onset.
        # Without this, the first consonant/syllable gets cut off.
        self._pre_buffer_size = 4  # ~128ms at 8kHz (4 x 256-sample chunks)
        self._pre_buffer: list[np.ndarray] = []

    def feed(self, audio_chunk: np.ndarray) -> str:
        """
        Feed 8kHz float32 audio (any size). Buffers internally to 256-sample
        chunks for Silero, returns the latest state.

        Returns:
            "speech"        — active speech detected
            "silence"       — no speech (or not enough yet)
            "end_of_speech" — caller finished speaking
        """
        self._pending = np.concatenate([self._pending, audio_chunk])

        result = "silence"
        while len(self._pending) >= SILERO_CHUNK:
            chunk = self._pending[:SILERO_CHUNK]
            self._pending = self._pending[SILERO_CHUNK:]
            result = self._process_chunk(chunk)
            if result == "end_of_speech":
                return result

        return result

    def _process_chunk(self, chunk: np.ndarray) -> str:
        """Run Silero VAD on a single 256-sample chunk."""
        tensor = torch.from_numpy(chunk).float()
        prob = self._model(tensor, SAMPLE_RATE_8K).item()
        is_speech = prob > self.speech_threshold

        if is_speech:
            if not self._speaking:
                # Speech just started — flush pre-buffer to capture onset
                self._buffer.extend(self._pre_buffer)
                self._pre_buffer.clear()
            self._speaking = True
            self._silent_count = 0
            self._speech_count += 1
            self._buffer.append(chunk)
            return "speech"

        if not self._speaking:
            # Not speaking — rotate pre-buffer (circular)
            self._pre_buffer.append(chunk)
            if len(self._pre_buffer) > self._pre_buffer_size:
                self._pre_buffer.pop(0)

        if self._speaking:
            self._silent_count += 1
            self._buffer.append(chunk)
            if self._silent_count >= self._silence_chunks:
                if self._speech_count >= self._min_speech_chunks:
                    return "end_of_speech"
                # Too short — noise blip, not real speech
                self.reset()
                return "silence"
            return "silence"

        return "silence"

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Quick check: is this chunk speech? Used for barge-in detection."""
        # Use a small buffer to check
        if len(audio_chunk) < SILERO_CHUNK:
            # Pad to minimum size
            padded = np.zeros(SILERO_CHUNK, dtype=np.float32)
            padded[:len(audio_chunk)] = audio_chunk
            audio_chunk = padded
        tensor = torch.from_numpy(audio_chunk[:SILERO_CHUNK]).float()
        prob = self._model(tensor, SAMPLE_RATE_8K).item()
        return prob > self.speech_threshold

    @property
    def speaking(self) -> bool:
        """Whether the VAD is currently tracking active speech."""
        return self._speaking

    def get_audio(self) -> np.ndarray:
        """Return accumulated speech audio buffer."""
        if not self._buffer:
            return np.array([], dtype=np.float32)
        return np.concatenate(self._buffer)

    def reset(self):
        """Clear buffer for next utterance."""
        self._buffer.clear()
        self._pre_buffer.clear()
        self._pending = np.array([], dtype=np.float32)
        self._silent_count = 0
        self._speech_count = 0
        self._speaking = False
        self._model.reset_states()
