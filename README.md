# HAL Answering Service

**A fully local AI phone screener.** Answers your calls as HAL 9000, collects the caller's name and reason, records the call, and sends you a push notification with a summary and full transcript.

No cloud AI. No per-minute billing. Runs entirely on your hardware using a local LLM, local speech-to-text, and local text-to-speech with voice cloning.

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
- **Customizable personality** -- HAL 9000 is just the default. Change the system prompt in `prompts.py` and swap the voice WAV file to create any character you want

## Example call

Listen to a real recorded call: [`recordings/example_call.wav`](recordings/example_call.wav)

## Latency

End-to-end latency from the caller finishing their sentence to hearing the AI respond runs under 1.5 seconds in typical conversation. Best observed round-trip is **649ms**, fully local, with voice cloning.

### Benchmarks (RTX 5090, glm-4.7-flash, Whisper large-v3-turbo)

Measured across 15 conversational exchanges over 3 live phone calls:

| Stage | Best | Typical | Worst |
|---|---|---|---|
| STT (Faster-Whisper) | 63 ms | 200--300 ms | 424 ms |
| LLM (time to first sentence) | 162 ms | 180--280 ms | 846 ms |
| TTS (Chatterbox Turbo, first chunk) | 345 ms | 500--850 ms | 1560 ms |
| **End-to-end** | **649 ms** | **~1.0--1.5 s** | **~2.8 s** |

> **Note:** These times start from when the VAD detects the caller has stopped speaking (after the 400ms silence threshold). The "worst" numbers are from the first exchange of a call when caches are cold. After that first turn, it's consistently faster.

### Why it feels fast

- **Sentence-level streaming** -- The LLM streams its response and TTS synthesizes each sentence as it arrives. The caller hears the first sentence while the rest is still being generated in the background.
- **Pre-recorded greetings** -- The initial pickup is instant. Greetings are synthesized at startup and played from memory, so there's zero TTS delay on the first thing the caller hears.
- **Barge-in** -- If the caller interrupts, audio clears instantly and the pipeline restarts. No waiting for the AI to finish its thought.
- **GPU concurrency** -- STT, LLM, and TTS all run on GPU. With enough VRAM, all three models stay loaded and hot.

### Hardware used for these benchmarks

| Component | Details |
|---|---|
| GPU | NVIDIA RTX 5090 (32 GB VRAM) |
| LLM | zai-org/glm-4.7-flash via LM Studio (thinking disabled) |
| STT | Faster-Whisper large-v3-turbo (float16) |
| TTS | Chatterbox Turbo with HAL 9000 voice clone |
| VAD | Silero VAD (400ms silence threshold) |

## Requirements

| Requirement | Details |
|---|---|
| **Python** | 3.12+ |
| **GPU** | NVIDIA with CUDA (Whisper + Chatterbox + VAD all run on GPU) |
| **VRAM** | 16 GB+ recommended (Whisper large-v3-turbo + Chatterbox Turbo + Silero VAD ≈ 6 GB, plus your LLM) |
| **SignalWire** | Account with a phone number ([signalwire.com](https://signalwire.com)) -- $0.50/mo for a number, ~$0.007/min for inbound calls |
| **Local LLM** | [LM Studio](https://lmstudio.ai) or any OpenAI-compatible API server |
| **Public endpoint** | HTTPS -- via Tailscale Funnel, Cloudflare Tunnel, ngrok, etc. |

## Quick start

### Windows

```powershell
git clone https://github.com/ninjahuttjr/hal-answering-service.git
cd hal-answering-service
setup.bat                # auto-detects your GPU and installs everything
copy .env.example .env   # then edit .env with your settings
python main.py
```

### Linux / macOS

```bash
git clone https://github.com/ninjahuttjr/hal-answering-service.git
cd hal-answering-service
chmod +x setup.sh
./setup.sh               # auto-detects your GPU and installs everything
cp .env.example .env     # then edit .env with your settings
python main.py
```

The setup script creates a virtual environment, detects your CUDA version, installs PyTorch with the correct CUDA index, and handles all dependencies. On first run, models download automatically (~3 GB).

You can force a specific CUDA version: `setup.bat cu124` / `./setup.sh cu124`, or use `cpu` for CPU-only.

## Setup

### 1. Install dependencies

Run the setup script for your platform:

**Windows:**
```powershell
setup.bat
```

**Linux / macOS:**
```bash
chmod +x setup.sh
./setup.sh
```

The setup script will:
- Create a Python virtual environment
- Auto-detect your NVIDIA GPU and install the right PyTorch + CUDA
- Install `chatterbox-tts` with `--no-deps` (the PyPI package has [broken dependency pins](https://github.com/resemble-ai/chatterbox/issues) for Python 3.12+)
- Install all other dependencies
- Verify that everything imports correctly

Override CUDA version if needed:

| Command | CUDA version |
|---|---|
| `setup.bat cu128` / `./setup.sh cu128` | CUDA 12.8 (RTX 50-series) |
| `setup.bat cu124` / `./setup.sh cu124` | CUDA 12.4 (RTX 40-series) |
| `setup.bat cu121` / `./setup.sh cu121` | CUDA 12.1 |
| `setup.bat cpu` / `./setup.sh cpu` | CPU only (no GPU) |

<details>
<summary>Manual install (without setup script)</summary>

**Windows:**
```powershell
python -m venv venv
venv\Scripts\activate

# Install PyTorch with CUDA (see https://pytorch.org/get-started/locally/)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install chatterbox-tts without its broken deps (must be >=0.1.5)
pip install --no-deps "chatterbox-tts>=0.1.5"

# Install everything else
pip install -r requirements.txt
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA (see https://pytorch.org/get-started/locally/)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install chatterbox-tts without its broken deps (must be >=0.1.5)
pip install --no-deps 'chatterbox-tts>=0.1.5'

# Install everything else
pip install -r requirements.txt
```

**Important:** `chatterbox-tts` must be version 0.1.5 or later. Earlier versions are missing the `chatterbox.tts_turbo` module and will crash on startup.

</details>

### 2. Configure

Copy the example config and fill in your values:

**Windows:** `copy .env.example .env`
**Linux / macOS:** `cp .env.example .env`

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
| `HF_TOKEN` | Hugging Face token -- needed to download the Chatterbox model on first run ([get one here](https://huggingface.co/settings/tokens)) |
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

Activate your virtual environment first, then start the server:

**Windows:**
```powershell
venv\Scripts\activate
python main.py
```

**Linux / macOS:**
```bash
source venv/bin/activate
python main.py
```

You should see the models load (~15-20s on first run), then:

```
HAL Answering Service ready in 16.8s -- listening on 0.0.0.0:8080
```

Call your SignalWire number and HAL will pick up.

## Voice cloning

Chatterbox Turbo can clone any voice from a short reference recording:

1. Record a **5+ second** WAV of the target voice (clean audio, minimal background noise).
2. Set `TTS_VOICE_PROMPT=/path/to/your/sample.wav` in `.env`.
3. Restart the server.

Two sample voices are included: `hal9000.wav` and `eugene.wav`.

## Call forwarding setup

You probably don't want every call going to an AI. Here's how I use it:

**Conditional forwarding (Verizon):** Dial `*71` followed by your SignalWire number. This only forwards calls you don't answer, so if you don't pick up it goes to HAL instead of voicemail. Other carriers have similar codes -- check your carrier's conditional forwarding instructions.

**Unknown callers only (iPhone):** Set up a Focus Mode that silences unknown numbers. Those calls go straight to HAL automatically while known contacts still ring through normally.

These two together mean: known contacts ring your phone like normal, and if you don't answer they get HAL. Unknown numbers skip the ring entirely and go straight to HAL.

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
| `LLM_TEMPERATURE` | `1.0` | Sampling temperature |
| **TTS** | | |
| `HF_TOKEN` | *(none)* | Hugging Face token for model download |
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

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'chatterbox.tts_turbo'` | `chatterbox-tts` is too old. Run: `pip install --no-deps "chatterbox-tts>=0.1.5"` |
| PyTorch CUDA not available after install | `chatterbox-tts` overwrote your CUDA PyTorch. Reinstall: `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124` |
| `pkg_resources` deprecation warning | Harmless. Comes from `resemble-perth`. Can be ignored. |
| Models download on every start | Set `HF_TOKEN` in `.env` so Hugging Face caches properly. First run downloads ~3 GB. |
| `CUDA out of memory` | Use a smaller STT model (`STT_MODEL=base`) or lower precision (`STT_COMPUTE_TYPE=int8`). |
| Connection refused on port 1234 | Start LM Studio (or your LLM server) before running `python main.py`. |

## Acknowledgments

- [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) by Resemble AI -- voice cloning
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) -- CTranslate2 Whisper implementation
- [Silero VAD](https://github.com/snakers4/silero-vad) -- voice activity detection
- [SignalWire](https://signalwire.com) -- telephony
- [LM Studio](https://lmstudio.ai) -- local LLM server
- [ntfy](https://ntfy.sh) -- push notifications

## License

[MIT](LICENSE)
