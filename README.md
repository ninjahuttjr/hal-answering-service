# HAL Answering Service

> A fully local AI phone screener that answers your calls as HAL 9000 -- collects the caller's name and reason, records the call, and sends you a summary.

No cloud AI. No per-minute billing. Everything runs on your hardware with a local LLM, local speech-to-text, and local text-to-speech with voice cloning.

## How it works

```
                          Your server (GPU)
                    +--------------------------+
Incoming call       |                          |
(SignalWire) ------>| WebSocket media stream   |
                    |   |                      |
                    |   v                      |
                    | Silero VAD               |
                    |   | (speech detected)    |
                    |   v                      |
                    | Faster-Whisper STT       |
                    |   | (text)               |
                    |   v                      |
                    | Local LLM (LM Studio)    |
                    |   | (response text)      |
                    |   v                      |
                    | Chatterbox TTS           |
                    |   | (audio)              |
Audio to caller <---|---+                      |
                    |                          |
                    +--------------------------+
```

1. Call arrives via [SignalWire](https://signalwire.com), forwarded as a real-time WebSocket stream.
2. **Silero VAD** detects when the caller starts and stops speaking.
3. **Faster-Whisper** transcribes the speech on GPU.
4. A **local LLM** (via [LM Studio](https://lmstudio.ai) or any OpenAI-compatible server) generates a response as HAL 9000.
5. **Chatterbox Turbo** synthesizes the response with voice cloning.
6. Audio is encoded as G.711 mu-law and streamed back to the caller.
7. When the call ends, a summary and transcript are pushed via [ntfy](https://ntfy.sh).

## Features

- **Fully local** -- no cloud AI APIs, everything on your GPU
- **Sub-second latency** -- real-time conversational AI on live phone calls
- **Voice cloning** -- clone any voice from a 5-second WAV sample (Chatterbox Turbo)
- **Barge-in** -- caller can interrupt the AI mid-sentence, audio clears instantly
- **Call recording** -- saves mixed mono WAV of both parties, accurately aligned
- **Push notifications** -- call summary + full transcript via ntfy
- **Pre-recorded greetings** -- instant pickup, no TTS delay
- **Silence handling** -- prompts quiet callers, auto-hangs-up after repeated silence
- **Security** -- webhook signature validation, input truncation, XML escaping, prompt injection hardening

## Requirements

| Requirement | Details |
|---|---|
| **Python** | 3.12+ |
| **GPU** | NVIDIA with CUDA (Whisper + Chatterbox + VAD all run on GPU) |
| **VRAM** | ~6 GB minimum (Whisper large-v3-turbo + Chatterbox Turbo + Silero VAD) |
| **SignalWire** | Account with a phone number ([signalwire.com](https://signalwire.com)) |
| **Local LLM** | [LM Studio](https://lmstudio.ai) or any OpenAI-compatible API server |
| **Public endpoint** | HTTPS -- via Tailscale Funnel, Cloudflare Tunnel, ngrok, etc. |

## Quick start

```bash
# Clone
git clone https://github.com/ninjahuttjr/hal-answering-service.git
cd hal-answering-service

# Install PyTorch with CUDA first (see https://pytorch.org/get-started/locally/)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install chatterbox-tts without its broken deps, then install everything
pip install --no-deps chatterbox-tts
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your SignalWire credentials, public hostname, etc.

# Start your local LLM in LM Studio, then:
python main.py
```

On first run, models download automatically (~3 GB). Subsequent starts use cached models.

> **Note:** `chatterbox-tts` on PyPI pins `numpy<1.26` which has no Python 3.12 wheels. Install it with `--no-deps` first -- `requirements.txt` includes all its actual runtime dependencies.

## Setup

### 1. Install dependencies

```bash
git clone https://github.com/ninjahuttjr/hal-answering-service.git
cd hal-answering-service
```

Install PyTorch with CUDA support for your system. See [pytorch.org](https://pytorch.org/get-started/locally/) for the right command. Example for CUDA 12.4:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Then install chatterbox-tts without its broken dependency constraints, and install everything else:

```bash
pip install --no-deps chatterbox-tts
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` -- the required fields are:

| Variable | What it is |
|---|---|
| `SIGNALWIRE_PROJECT_ID` | From your SignalWire dashboard |
| `SIGNALWIRE_TOKEN` | API token from SignalWire |
| `SIGNALWIRE_SPACE` | Your SignalWire space name |
| `SIGNALWIRE_PHONE_NUMBER` | The number that receives calls |
| `PUBLIC_HOST` | Your public HTTPS hostname (see step 4) |
| `OWNER_NAME` | Name HAL uses when referring to you |

Optional but recommended:

| Variable | What it is |
|---|---|
| `HF_TOKEN` | Hugging Face token for downloading Chatterbox Turbo ([get one here](https://huggingface.co/settings/tokens)) |
| `TTS_VOICE_PROMPT` | Path to a WAV file (>5s) for voice cloning |
| `NTFY_TOPIC` | [ntfy.sh](https://ntfy.sh) topic for call notifications |
| `SIGNALWIRE_SIGNING_KEY` | Webhook signing key (falls back to `SIGNALWIRE_TOKEN` if unset) |

See `.env.example` for the full list of options with defaults.

### 3. Start your local LLM

Open [LM Studio](https://lmstudio.ai) and load a model. The default config expects the server at `http://127.0.0.1:1234/v1`. Any OpenAI-compatible API works.

Recommended models: anything fast with good instruction following. The default is `zai-org/glm-4.7-flash`.

### 4. Expose your server

You need a public HTTPS endpoint forwarding to port 8080:

```bash
# Tailscale Funnel (easiest if you use Tailscale)
tailscale funnel 8080

# Cloudflare Tunnel
cloudflared tunnel --url http://localhost:8080

# ngrok
ngrok http 8080
```

Set `PUBLIC_HOST` in `.env` to the hostname you get (without `https://`).

### 5. Configure SignalWire

In the [SignalWire dashboard](https://signalwire.com), set your phone number's **incoming call webhook** to:

```
https://YOUR_PUBLIC_HOST/incoming-call
```

Make sure it's set to **POST** and the format is **XML**.

### 6. Run

```bash
python main.py
```

You should see the models load (~15-20s on first run), then:

```
SuperCaller ready in 16.8s -- listening on 0.0.0.0:8080
```

Call your SignalWire number and HAL will pick up.

## Voice cloning

Chatterbox Turbo can clone any voice from a short reference recording:

1. Record a **5+ second** WAV of the target voice (clean audio, minimal background noise).
2. Set `TTS_VOICE_PROMPT=/path/to/your/sample.wav` in `.env`.
3. Restart the server.

Two sample voices are included: `hal9000.wav` and `eugene.wav`.

## Project structure

| File | Purpose |
|---|---|
| `main.py` | Entry point -- loads models, pre-records greetings, starts server |
| `server.py` | FastAPI app -- webhook, WebSocket media stream, ntfy notifications |
| `call_handler.py` | Per-call pipeline -- VAD, STT, LLM, TTS, barge-in, recording |
| `audio.py` | G.711 mu-law codec, resampling, Silero VAD wrapper |
| `stt.py` | Faster-Whisper speech-to-text |
| `tts.py` | Chatterbox Turbo text-to-speech with voice cloning |
| `llm.py` | OpenAI-compatible LLM client with streaming sentence extraction |
| `prompts.py` | HAL 9000 system prompt, greetings, summary prompt |
| `config.py` | Dataclass config from environment variables |

## Configuration reference

All settings are configured via environment variables (`.env` file).

<details>
<summary>Full variable list</summary>

| Variable | Default | Description |
|---|---|---|
| **SignalWire** | | |
| `SIGNALWIRE_PROJECT_ID` | *(required)* | Project ID from dashboard |
| `SIGNALWIRE_TOKEN` | *(required)* | API token |
| `SIGNALWIRE_SPACE` | *(required)* | Space name |
| `SIGNALWIRE_PHONE_NUMBER` | *(required)* | Phone number for inbound calls |
| `SIGNALWIRE_SIGNING_KEY` | *(uses token)* | Webhook signing key |
| **Server** | | |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8080` | Bind port |
| `PUBLIC_HOST` | *(required)* | Public hostname for WebSocket URL |
| `MAX_CONCURRENT_CALLS` | `3` | Max simultaneous calls |
| `MAX_CALL_DURATION_S` | `600` | Max call length in seconds |
| **STT** | | |
| `STT_MODEL` | `large-v3-turbo` | Faster-Whisper model size |
| `STT_DEVICE` | `cuda` | `cuda` or `cpu` |
| `STT_COMPUTE_TYPE` | `float16` | `float16`, `int8`, etc. |
| **LLM** | | |
| `LLM_BASE_URL` | `http://127.0.0.1:1234/v1` | OpenAI-compatible API endpoint |
| `LLM_API_KEY` | `lm-studio` | API key (LM Studio ignores this) |
| `LLM_MODEL` | `zai-org/glm-4.7-flash` | Model name |
| `LLM_MAX_TOKENS` | `200` | Max response tokens |
| `LLM_TEMPERATURE` | `0.7` | Sampling temperature |
| **TTS** | | |
| `TTS_VOICE_PROMPT` | *(none)* | Path to voice cloning WAV (>5s) |
| **VAD** | | |
| `VAD_SPEECH_THRESHOLD` | `0.5` | Silero speech probability threshold |
| `VAD_SILENCE_THRESHOLD_MS` | `400` | Silence duration to end utterance |
| `VAD_MIN_SPEECH_MS` | `250` | Min speech duration to be real |
| **Other** | | |
| `OWNER_NAME` | *(none)* | Your name (used in greetings) |
| `RECORDINGS_DIR` | `recordings` | Where call WAVs are saved |
| `NTFY_TOPIC` | *(none)* | ntfy.sh topic for notifications |

</details>

## Acknowledgments

- [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) by Resemble AI -- voice cloning
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) -- CTranslate2 Whisper implementation
- [Silero VAD](https://github.com/snakers4/silero-vad) -- voice activity detection
- [SignalWire](https://signalwire.com) -- telephony
- [LM Studio](https://lmstudio.ai) -- local LLM server
- [ntfy](https://ntfy.sh) -- push notifications

## License

[MIT](LICENSE)
