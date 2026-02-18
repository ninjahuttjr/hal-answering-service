"""Speech-to-text for telephony audio using Faster-Whisper."""

import logging
import threading
import numpy as np
from faster_whisper import WhisperModel

from audio import resample

log = logging.getLogger(__name__)

WHISPER_SAMPLE_RATE = 16000


class SpeechToText:
    """Faster-Whisper STT wrapper optimized for telephony speed."""

    def __init__(self, model_size: str = "medium.en", device: str = "cuda",
                 compute_type: str = "float16"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model: WhisperModel | None = None
        self._lock = threading.Lock()  # Protect model state from concurrent calls

    def load(self):
        """Initialize the Whisper model."""
        log.info("Loading Faster-Whisper model: %s on %s (%s)",
                 self.model_size, self.device, self.compute_type)
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        log.info("Faster-Whisper model loaded.")

    def transcribe(self, audio: np.ndarray, input_sr: int = 8000) -> str:
        """
        Transcribe audio to text. Optimized for low-latency telephony.
        """
        if self.model is None:
            raise RuntimeError("STT model not loaded. Call load() first.")

        if input_sr <= 0:
            raise ValueError(f"input_sr must be positive, got {input_sr}")

        if len(audio) == 0:
            return ""

        # Resample to 16kHz for Whisper
        if input_sr != WHISPER_SAMPLE_RATE:
            audio = resample(audio, input_sr, WHISPER_SAMPLE_RATE)

        # Pad short audio with silence — Whisper hallucinates on clips < 1s.
        # 0.5s of silence on each side gives the decoder stable context.
        min_samples = int(WHISPER_SAMPLE_RATE * 1.5)  # 1.5s minimum
        if len(audio) < min_samples:
            pad_size = int(WHISPER_SAMPLE_RATE * 0.5)  # 0.5s padding
            audio = np.concatenate([
                np.zeros(pad_size, dtype=np.float32),
                audio,
                np.zeros(pad_size, dtype=np.float32),
            ])

        with self._lock:
            segments, info = self.model.transcribe(
                audio,
                beam_size=1,                    # Greedy decode — fastest
                best_of=1,                      # No sampling alternatives
                vad_filter=False,               # OFF — Silero VAD already isolated speech
                no_speech_threshold=0.6,        # Skip segments with high no-speech probability
                log_prob_threshold=-1.0,        # Relaxed — don't drop valid short utterances
                condition_on_previous_text=False,  # Faster, no cascading errors
                language="en",                  # Skip language detection
                initial_prompt="Phone call screening conversation.",  # Prime decoder for telephony context
            )

            text_parts = []
            for segment in segments:
                text_parts.append(segment.text.strip())

        result = " ".join(text_parts).strip()
        if result:
            lang_prob = getattr(info, 'language_probability', None)
            if lang_prob is not None:
                log.info("STT: %s (prob=%.2f)", result, lang_prob)
            else:
                log.info("STT: %s", result)
        return result
