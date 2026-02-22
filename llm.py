"""OpenAI-compatible LLM client — streams responses and yields complete sentences for TTS."""

import re
import logging
from typing import Generator

import httpx
from openai import OpenAI

from config import Config
from prompts import build_system_prompt, build_summary_prompt

log = logging.getLogger(__name__)

# Sentence boundary pattern — split on . ! ? followed by space or end
SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+')

MAX_HISTORY_MESSAGES = 100  # 50 turns (user + assistant) — scammer calls need long context
MAX_SENTENCES_PER_RESPONSE = 6  # Phone calls: 1-2 normal, 6 is generous safety cap
SUPPORTED_LLM_PROVIDERS = {"auto", "lmstudio", "ollama", "openai_compatible"}

# Timeout for LLM requests — prevents indefinite hangs if LLM server is unresponsive
LLM_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)

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


def _normalize_provider(value: str) -> str:
    provider = (value or "auto").strip().lower()
    return provider if provider in SUPPORTED_LLM_PROVIDERS else "auto"


def _infer_provider(base_url: str) -> str:
    url = (base_url or "").strip().lower()
    if "127.0.0.1:1234" in url or "localhost:1234" in url:
        return "lmstudio"
    if "127.0.0.1:11434" in url or "localhost:11434" in url or "ollama" in url:
        return "ollama"
    return "openai_compatible"


def _looks_like_extra_body_issue(exc: Exception) -> bool:
    text = str(exc).lower()
    hints = (
        "chat_template_kwargs",
        "enable_thinking",
        "extra_body",
        "unknown field",
        "unknown parameter",
        "invalid parameter",
    )
    return any(h in text for h in hints)


def _sanitize(text: str) -> str:
    """Strip known garbage tokens and leaked role prefixes from LLM output."""
    cleaned = _GARBAGE_RE.sub('', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    # Strip leaked role prefixes (e.g. "HAL:", "Caller:", "Assistant:") that
    # smaller LLMs copy from few-shot examples or conversation history.
    cleaned = re.sub(r'^(?:HAL|Caller|Assistant)\s*:\s*', '', cleaned, flags=re.IGNORECASE).strip()
    return cleaned


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
        requested = _normalize_provider(config.llm_provider)
        self.provider = _infer_provider(config.llm_base_url) if requested == "auto" else requested
        self._client_base_url = config.llm_base_url
        self._client_api_key = config.llm_api_key
        self.client = OpenAI(
            base_url=self._client_base_url,
            api_key=self._client_api_key,
            timeout=LLM_TIMEOUT,
        )
        self.history: list[dict] = []
        log.info("LLM provider: %s (base_url=%s)", self.provider, config.llm_base_url)

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
        # Trim to max history — drop oldest messages to stay within context limits
        if len(self.history) > MAX_HISTORY_MESSAGES:
            dropped = len(self.history) - MAX_HISTORY_MESSAGES
            self.history = self.history[-MAX_HISTORY_MESSAGES:]
            log.info("History trimmed: dropped %d oldest messages (%d remaining)", dropped, len(self.history))

    def _build_messages(self) -> list[dict]:
        """Build the full message list with system prompt."""
        system_prompt = build_system_prompt(owner_name=self.config.owner_name)
        return [{"role": "system", "content": system_prompt}] + self.history

    def _refresh_client_if_needed(self):
        """Recreate OpenAI client if base URL or API key changed at runtime."""
        desired_url = self.config.llm_base_url
        desired_key = self.config.llm_api_key
        if desired_url == self._client_base_url and desired_key == self._client_api_key:
            return
        self._client_base_url = desired_url
        self._client_api_key = desired_key
        self.client = OpenAI(base_url=desired_url, api_key=desired_key, timeout=LLM_TIMEOUT)
        requested = _normalize_provider(self.config.llm_provider)
        self.provider = _infer_provider(desired_url) if requested == "auto" else requested
        log.info("LLM client reloaded (provider=%s, base_url=%s)", self.provider, desired_url)

    def _build_chat_kwargs(self, messages: list[dict], stream: bool, max_tokens: int,
                           temperature: float) -> dict:
        kwargs = dict(
            model=self.config.llm_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )
        if self.config.llm_frequency_penalty:
            kwargs["frequency_penalty"] = self.config.llm_frequency_penalty
        # LM Studio-specific flag to disable thinking/reasoning mode.
        if self.provider == "lmstudio":
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        return kwargs

    def _create_chat_completion(self, messages: list[dict], stream: bool, max_tokens: int,
                                temperature: float):
        kwargs = self._build_chat_kwargs(
            messages=messages,
            stream=stream,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        try:
            return self.client.chat.completions.create(**kwargs)
        except Exception as e:
            if self.provider == "lmstudio" and "extra_body" in kwargs and _looks_like_extra_body_issue(e):
                retry_kwargs = dict(kwargs)
                retry_kwargs.pop("extra_body", None)
                log.warning("LLM backend rejected LM Studio extras; retrying without extra_body")
                return self.client.chat.completions.create(**retry_kwargs)
            raise

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
        self._refresh_client_if_needed()

        log.info("LLM request: %d messages, last user: %s", len(messages), text[:80])

        try:
            stream = self._create_chat_completion(
                messages=messages,
                stream=True,
                max_tokens=self.config.llm_max_tokens,
                temperature=self.config.llm_temperature,
            )
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
        self._refresh_client_if_needed()
        messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes phone calls."},
            {"role": "user", "content": prompt},
        ]

        response = self._create_chat_completion(
            messages=messages,
            stream=False,
            max_tokens=300,
            temperature=0.3,
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
