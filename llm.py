"""OpenAI-compatible LLM client — streams responses and yields complete sentences for TTS."""

import re
import logging
from typing import Generator

from openai import OpenAI

from config import Config
from prompts import build_system_prompt, build_summary_prompt

log = logging.getLogger(__name__)

# Sentence boundary pattern — split on . ! ? followed by space or end
SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+')

MAX_HISTORY_MESSAGES = 20  # 10 turns (user + assistant)
MAX_SENTENCES_PER_RESPONSE = 6  # Phone calls: 1-2 normal, 6 is generous safety cap

# GLM-4 special tokens and other garbage that can leak from LLMs
_GARBAGE_RE = re.compile(
    r'\[g?MASK\]'               # GLM mask tokens
    r'|\[CLS\]|\[SEP\]|\[PAD\]' # BERT-style tokens
    r'|</?[a-z][a-z0-9]*[^>]*>' # HTML tags
    r'|###[^\n]*'               # Leaked markdown headers
    r'|```[^\n]*'               # Code fences
    r'|~+</?[a-z]+'             # Malformed tags (e.g. ~</b)
    r'|</?think>'               # GLM thinking tags that leak through
    , re.IGNORECASE
)


def _sanitize(text: str) -> str:
    """Strip known garbage tokens from LLM output."""
    cleaned = _GARBAGE_RE.sub('', text)
    return re.sub(r'\s+', ' ', cleaned).strip()


def _has_real_words(text: str) -> bool:
    """Check that text contains at least one real word (2+ letters)."""
    return bool(re.search(r'[a-zA-Z]{2,}', text))


def _is_degenerate(text: str) -> bool:
    """Detect repetitive/nonsensical LLM output (e.g. word loops)."""
    words = re.findall(r'[a-zA-Z]{2,}', text.lower())
    if len(words) >= 6 and len(set(words)) / len(words) < 0.35:
        return True
    return False


def trim_to_complete_sentence(text: str) -> str:
    """Truncate text at the last complete sentence boundary."""
    text = text.strip()
    if not text:
        return text
    # Find the last sentence-ending punctuation
    for i in range(len(text) - 1, -1, -1):
        if text[i] in '.!?':
            return text[:i + 1]
    return text


def extract_sentences(text: str) -> list[str]:
    """Split text into sentences on . ! ? boundaries."""
    parts = SENTENCE_BOUNDARY.split(text)
    return [s.strip() for s in parts if s.strip()]


class LLMClient:
    """OpenAI-compatible LLM client for call screening conversations."""

    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
        )
        self.history: list[dict] = []

    def reset_history(self):
        """Clear conversation history."""
        self.history.clear()

    def add_user_message(self, text: str):
        """Append a user message to history.

        Prefixes with 'Caller:' to help smaller LLMs maintain clear
        speaker identity (prevents the model from confusing itself
        with the caller when the caller says things like 'I am Dave').
        """
        self.history.append({"role": "user", "content": f"Caller: {text}"})
        # Trim to max history
        if len(self.history) > MAX_HISTORY_MESSAGES:
            self.history = self.history[-MAX_HISTORY_MESSAGES:]

    def _build_messages(self) -> list[dict]:
        """Build the full message list with system prompt."""
        system_prompt = build_system_prompt(owner_name=self.config.owner_name)
        return [{"role": "system", "content": system_prompt}] + self.history

    def chat_stream_sentences(self, text: str) -> Generator[tuple[str, bool], None, None]:
        """
        Send user text to LLM and yield (sentence, is_final) tuples
        as complete sentences arrive from the stream.

        History management is NOT done here — the caller (CallHandler)
        is responsible for adding user/assistant messages to history,
        since it knows what was actually spoken vs. interrupted.

        Args:
            text: User's transcribed speech.

        Yields:
            (sentence, is_final) — sentence text and whether it's the last one.
        """
        messages = self._build_messages()

        log.info("LLM request: %d messages, last user: %s", len(messages), text[:80])

        try:
            kwargs = dict(
                model=self.config.llm_model,
                messages=messages,
                max_tokens=self.config.llm_max_tokens,
                temperature=self.config.llm_temperature,
                stream=True,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            if self.config.llm_frequency_penalty:
                kwargs["frequency_penalty"] = self.config.llm_frequency_penalty
            stream = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            log.error("LLM stream creation failed: %s", e)
            return

        buffer = ""
        full_response = ""
        sentences_yielded = []
        garbage_count = 0
        degenerate = False
        max_reached = False

        try:
            for chunk in stream:
                delta = chunk.choices[0].delta
                # Fallback: if thinking mode is stuck on, use reasoning_content as content
                has_reasoning = hasattr(delta, 'reasoning_content') and delta.reasoning_content
                token = delta.content
                if not token and has_reasoning:
                    # LM Studio thinking mode is active — reasoning_content is all we get
                    token = delta.reasoning_content
                    log.debug("Using reasoning_content as fallback: %s", token[:80])
                if token:
                    buffer += token
                    full_response += token

                    # Check for complete sentences in buffer
                    while True:
                        match = SENTENCE_BOUNDARY.search(buffer)
                        if match:
                            raw = buffer[:match.start()].strip()
                            buffer = buffer[match.end():]
                            sentence = _sanitize(raw)

                            if sentence and _has_real_words(sentence):
                                # Check full response for degeneration (repetitive loops)
                                if _is_degenerate(full_response):
                                    log.warning("LLM output degenerate, aborting stream")
                                    degenerate = True
                                    break
                                sentences_yielded.append(sentence)
                                yield (sentence, False)

                                if len(sentences_yielded) >= MAX_SENTENCES_PER_RESPONSE:
                                    log.info("Hit max sentences (%d), truncating cleanly",
                                                MAX_SENTENCES_PER_RESPONSE)
                                    max_reached = True
                                    break
                            elif raw:
                                garbage_count += 1
                                log.warning("Skipped garbage sentence: %s", raw[:80])
                                if garbage_count >= 3:
                                    log.warning("Too many garbage sentences, aborting")
                                    degenerate = True
                                    break
                        else:
                            break

                    if degenerate or max_reached:
                        break
        except Exception as e:
            log.error("LLM stream iteration error: %s", e)

        # Yield remaining buffer as final sentence (only if not truncated or degenerate)
        remaining = _sanitize(buffer)
        if remaining and _has_real_words(remaining) and not degenerate and not max_reached:
            if remaining[-1] not in '.!?':
                remaining += '.'
            sentences_yielded.append(remaining)
            yield (remaining, True)

        # Log the response (history is managed by CallHandler)
        if sentences_yielded:
            log.info("LLM response: %s", ' '.join(sentences_yielded)[:100])
            if degenerate:
                log.warning("LLM went degenerate — stopped early")
        else:
            log.warning("LLM produced no usable output")

    def truncate_last_response(self, spoken_text: str, was_interrupted: bool):
        """
        After barge-in, truncate the last assistant message to what was
        actually spoken, so the LLM's context matches what the caller heard.
        """
        if not was_interrupted or not self.history:
            return
        # Find the last assistant message
        for i in range(len(self.history) - 1, -1, -1):
            if self.history[i]["role"] == "assistant":
                original = self.history[i]["content"]
                truncated = trim_to_complete_sentence(spoken_text)
                if truncated:
                    self.history[i]["content"] = truncated + " [interrupted]"
                    log.info("Truncated response: %s -> %s", original[:50], truncated[:50])
                break

    def get_summary(self, transcript: list[dict]) -> str:
        """Generate a call summary using the LLM."""
        prompt = build_summary_prompt(transcript)
        messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes phone calls."},
            {"role": "user", "content": prompt},
        ]

        response = self.client.chat.completions.create(
            model=self.config.llm_model,
            messages=messages,
            max_tokens=300,
            temperature=0.3,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        result = response.choices[0].message
        content = result.content or ""
        # Fallback: if thinking mode consumed everything, use reasoning_content
        if not content.strip() and hasattr(result, 'reasoning_content') and result.reasoning_content:
            log.warning("Summary: content empty, falling back to reasoning_content")
            content = result.reasoning_content
            # Strip any <think> tags that leaked through
            content = re.sub(r'</?think>', '', content)
        return content.strip()
