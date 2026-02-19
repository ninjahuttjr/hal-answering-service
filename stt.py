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
                 compute_type: str = "float16",
                 language: str = "en",
                 beam_size: int = 1,
                 best_of: int = 1,
                 no_speech_threshold: float = 0.6,
                 log_prob_threshold: float = -1.0,
                 condition_on_previous_text: bool = False,
                 initial_prompt: str = "Phone call screening conversation."):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.best_of = best_of
        self.no_speech_threshold = no_speech_threshold
        self.log_prob_threshold = log_prob_threshold
        self.condition_on_previous_text = condition_on_previous_text
        self.initial_prompt = initial_prompt
        self.model: WhisperModel | None = None
        self._lock = threading.Lock()  # Protect model state from concurrent calls

    def load(self):
        """Initialize the Whisper model."""
        log.info(
            "Loading Faster-Whisper model: %s on %s (%s) lang=%s beam=%d best_of=%d",
            self.model_size,
            self.device,
            self.compute_type,
            self.language.strip() or "auto",
            self.beam_size,
            self.best_of,
        )
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
                beam_size=self.beam_size,
                best_of=self.best_of,
                vad_filter=False,               # OFF — Silero VAD already isolated speech
                no_speech_threshold=self.no_speech_threshold,
                log_prob_threshold=self.log_prob_threshold,
                condition_on_previous_text=self.condition_on_previous_text,
                language=self.language.strip() or None,
                initial_prompt=self.initial_prompt.strip() or None,
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
