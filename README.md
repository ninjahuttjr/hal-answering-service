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
                    | Local LLM (LM Studio / Ollama) |
                    |   | (response text)      |
                    |   v                      |
                    | Chatterbox TTS           |
                    |   | (audio)              |
Audio to caller <---|---+                      |
                    |                          |
                    +--------------------------+
```

## Features

- **Fully local** -- no cloud AI APIs, everything on your GPU
- **Sub-second latency** -- best observed round-trip is **649ms**, fully local, with voice cloning
- **Voice cloning** -- clone any voice from a 5-second WAV sample (Chatterbox Turbo)
- **Barge-in** -- caller can interrupt the AI mid-sentence, audio clears instantly
- **Call recording** -- saves mixed mono WAV of both parties, accurately aligned
- **Push notifications** -- call summary + full transcript via ntfy
- **Call metadata** -- JSON logs of every call with transcript and summary
- **Pre-recorded greetings** -- instant pickup, no TTS delay
- **Silence handling** -- prompts quiet callers, auto-hangs-up after repeated silence
- **Security** -- webhook signature validation, input truncation, XML escaping, prompt injection hardening
- **Customizable personality** -- HAL 9000 is just the default. Change the system prompt in `prompts.py` and swap the voice WAV file to create any character you want

## Example call

HAL vs. a warranty scammer -- keeps him on the line with increasingly specific questions until Dave gives up:

<video src="https://github.com/user-attachments/assets/7e27ed27-0f59-45ed-8797-0300f976b2f6" controls></video>

## Try it now (demo mode)

Talk to HAL through your browser -- no phone number, no accounts, no public server needed. Just a local LLM.

```bash
git clone https://github.com/ninjahuttjr/hal-answering-service.git
cd hal-answering-service
setup.bat               # Windows (or ./setup.sh on Linux/macOS)
python main.py --demo
```

On first run it will walk you through provider setup (`LM Studio`, `Ollama`, or another OpenAI-compatible server), then write a demo `.env`. After startup, open **http://localhost:8080/demo**. The demo UI includes readiness checks, a live config editor, and a real-time transcript display.

You need a local LLM server running before you start. Everything else -- speech-to-text, text-to-speech, voice activity detection -- runs locally with no cloud AI API.

> **Mic access:** Browsers require either `localhost` or HTTPS for microphone access. The demo auto-redirects LAN IPs to localhost. For remote access, use `--share` to create a public HTTPS tunnel.

> **Tip:** Use headphones to avoid echo feedback.

### Remote access with `--share`

```bash
python main.py --demo --share
```

This creates a public HTTPS URL (via Gradio's tunnel) so you can test from any device -- phone, tablet, another computer. Mic access works everywhere over HTTPS.

## Requirements

| Requirement | Details |
|---|---|
| **Python** | 3.12+ |
| **GPU** | Optional: NVIDIA CUDA recommended for best latency. CPU mode is supported (including macOS). |
| **VRAM** | 16 GB+ recommended for full GPU stack (Whisper large-v3-turbo + Chatterbox Turbo + Silero VAD ~ 6 GB, plus your LLM) |
| **SignalWire** | Account with a phone number ([signalwire.com](https://signalwire.com)) -- $0.50/mo for a number, ~$0.007/min for inbound calls. **Not needed for demo mode.** |
| **Local LLM** | [LM Studio](https://lmstudio.ai), [Ollama](https://ollama.com), or any OpenAI-compatible API server |
| **Public endpoint** | HTTPS -- via Tailscale Funnel, Cloudflare Tunnel, ngrok, etc. **Not needed for demo mode.** |

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

The setup script creates a virtual environment, installs CUDA PyTorch when an NVIDIA GPU is present (or CPU PyTorch otherwise), and handles all dependencies.

You also need a local LLM running:

- **LM Studio:** default endpoint `http://127.0.0.1:1234/v1`
- **Ollama:** use OpenAI-compatible endpoint `http://127.0.0.1:11434/v1`

<details>
<summary>Detailed setup guide</summary>

### Install dependencies

Run the setup script for your platform. It will:
- Create a Python virtual environment
- Auto-detect your NVIDIA GPU and install the right PyTorch build (CUDA or CPU)
- Install `chatterbox-tts` with `--no-deps` (the PyPI package has [broken dependency pins](https://github.com/resemble-ai/chatterbox/issues) for Python 3.12+)
- Install all other dependencies
- Verify that everything imports correctly

Important: run setup as your normal user (no `sudo`). Running with `sudo` can create a root-owned `venv` and break later installs.

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

# Install PyTorch (macOS / CPU default):
pip install torch torchaudio

# If you're on NVIDIA Linux, use CUDA wheels instead:
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install chatterbox-tts without its broken deps (must be >=0.1.5)
pip install --no-deps 'chatterbox-tts>=0.1.5'

# Install everything else
pip install -r requirements.txt
```

**Important:** `chatterbox-tts` must be version 0.1.5 or later. Earlier versions are missing the `chatterbox.tts_turbo` module and will crash on startup.

</details>

### Configure

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
| `PUBLIC_HOST` | Your public HTTPS hostname (see below) |
| `OWNER_NAME` | Name HAL uses when referring to you |

Optional but recommended:

| Variable | What it is |
|---|---|
| `HF_TOKEN` | Optional Hugging Face token -- useful if anonymous model download fails or is rate-limited ([get one here](https://huggingface.co/settings/tokens)) |
| `TTS_MODEL_DIR` | Optional local directory with pre-downloaded Chatterbox weights (for fully offline onboarding) |
| `TTS_DEVICE` | `auto`, `cuda`, or `cpu` (default is `auto`) |
| `TTS_VOICE_PROMPT` | Path to a WAV file (>5s) for voice cloning |
| `NTFY_TOPIC` | [ntfy.sh](https://ntfy.sh) topic for call notifications |
| `SIGNALWIRE_SIGNING_KEY` | Webhook signing key (falls back to `SIGNALWIRE_TOKEN` if unset) |

See `.env.example` for the full list of options with defaults.

### Start your local LLM

Choose one:

- **LM Studio**: open [LM Studio](https://lmstudio.ai), load a model, and start the local server at `http://127.0.0.1:1234/v1`.
- **Ollama**: run:
  ```bash
  ollama pull qwen3:4b
  ollama serve
  ```
  then use `LLM_BASE_URL=http://127.0.0.1:11434/v1` and `LLM_PROVIDER=ollama`.
- **Other**: any OpenAI-compatible API endpoint is supported.

Recommended models: anything fast with good instruction following. The default is `zai-org/glm-4.7-flash`.

### Bundle Chatterbox Weights (offline onboarding)

To avoid runtime model downloads, ship a local model bundle and point HAL at it:

```bash
# Download once into a distributable local folder
python scripts/prefetch_chatterbox.py --output models/chatterbox
```

Set one of:

- `TTS_MODEL_DIR=models/chatterbox` in `.env`, or
- leave `TTS_MODEL_DIR` unset and place files in `models/chatterbox` (auto-detected).

Required files in the bundle directory:

- `ve.safetensors`
- `t3_turbo_v1.safetensors`
- `s3gen_meanflow.safetensors`
- tokenizer files (`tokenizer.json`, `tokenizer_config.json`, etc.)

When the bundle is present, HAL loads TTS fully local and does not call Hugging Face.

### Expose your server

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

### Configure SignalWire

In the [SignalWire dashboard](https://signalwire.com), set your phone number's **incoming call webhook** to:

```
https://YOUR_PUBLIC_HOST/incoming-call
```

Make sure it's set to **POST** and the format is **XML**.

</details>

## Call data

- Recordings are saved as WAV files in the `recordings/` directory
- Call metadata (transcript, summary, caller info) is stored as JSON in the `metadata/` directory
- `GET /health` returns `{"status": "ok"}` for uptime monitoring

## Voice cloning

Chatterbox Turbo can clone any voice from a short reference recording:

1. Record a **5+ second** WAV of the target voice (clean audio, minimal background noise).
2. Set `TTS_VOICE_PROMPT=/path/to/your/sample.wav` in `.env`.
3. Restart the server.

Two sample voices are included: `hal9000.wav` and `eugene.wav`.

<details>
<summary>Call forwarding setup</summary>

You probably don't want every call going to an AI. Here's how I use it:

**Conditional forwarding (Verizon):** Dial `*71` followed by your SignalWire number. This only forwards calls you don't answer, so if you don't pick up it goes to HAL instead of voicemail. Other carriers have similar codes -- check your carrier's conditional forwarding instructions.

**Unknown callers only (iPhone):** Set up a Focus Mode that silences unknown numbers. Those calls go straight to HAL automatically while known contacts still ring through normally.

These two together mean: known contacts ring your phone like normal, and if you don't answer they get HAL. Unknown numbers skip the ring entirely and go straight to HAL.

</details>

<details>
<summary>Latency benchmarks</summary>

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

</details>

## Project structure

| File | Purpose |
|---|---|
| `main.py` | Entry point -- loads models, pre-records greetings, starts server |
| `server.py` | FastAPI app -- webhook, WebSocket media stream, demo endpoints, ntfy notifications |
| `gradio_ui.py` | Gradio-based demo UI -- HAL eye, call controls, transcript, settings |
| `call_handler.py` | Per-call pipeline -- VAD, STT, LLM, TTS, barge-in, recording |
| `audio.py` | G.711 mu-law codec, resampling, Silero VAD wrapper |
| `stt.py` | Faster-Whisper speech-to-text |
| `tts.py` | Chatterbox Turbo text-to-speech with voice cloning |
| `llm.py` | OpenAI-compatible LLM client with streaming sentence extraction |
| `prompts.py` | HAL 9000 system prompt, greetings, summary prompt |
| `config.py` | Dataclass config from environment variables |

<details>
<summary>Configuration reference</summary>

All settings are configured via environment variables (`.env` file).

| Variable | Default | Description |
|---|---|---|
| **SignalWire** | | |
| `SIGNALWIRE_PROJECT_ID` | *(required)* | Project ID from dashboard |
| `SIGNALWIRE_TOKEN` | *(required)* | API token |
| `SIGNALWIRE_SPACE` | *(required)* | Space name |
| `SIGNALWIRE_PHONE_NUMBER` | *(required)* | Phone number for inbound calls |
| `SIGNALWIRE_SIGNING_KEY` | *(uses token)* | Webhook signing key |
| **Server** | | |
| `HOST` | `127.0.0.1` | Bind address (`0.0.0.0` for all interfaces) |
| `PORT` | `8080` | Bind port |
| `PUBLIC_HOST` | *(required)* | Public hostname for WebSocket URL |
| `MAX_CONCURRENT_CALLS` | `3` | Max simultaneous calls |
| `MAX_CALL_DURATION_S` | `600` | Max call length in seconds |
| **STT** | | |
| `STT_MODEL` | `large-v3-turbo` | Faster-Whisper model size |
| `STT_DEVICE` | `auto` | `auto`, `cuda`, or `cpu` |
| `STT_COMPUTE_TYPE` | `auto` | `auto`, `float16`, `int8`, etc. |
| `STT_LANGUAGE` | `en` | Language hint (`""` to auto-detect) |
| `STT_BEAM_SIZE` | `1` | Beam width (higher can improve accuracy, slower) |
| `STT_BEST_OF` | `1` | Number of sampled candidates |
| `STT_NO_SPEECH_THRESHOLD` | `0.6` | Probability threshold for no-speech |
| `STT_LOG_PROB_THRESHOLD` | `-1.0` | Minimum log-prob threshold |
| `STT_CONDITION_ON_PREVIOUS_TEXT` | `false` | Use previous text as decoding context |
| `STT_INITIAL_PROMPT` | `Phone call screening conversation.` | Initial prompt for decoder |
| **LLM** | | |
| `LLM_PROVIDER` | `auto` | `auto`, `lmstudio`, `ollama`, or `openai_compatible` |
| `LLM_BASE_URL` | `http://127.0.0.1:1234/v1` | OpenAI-compatible API endpoint |
| `LLM_API_KEY` | `lm-studio` | API key (LM Studio ignores this) |
| `LLM_MODEL` | `zai-org/glm-4.7-flash` | Model name |
| `LLM_MAX_TOKENS` | `200` | Max response tokens |
| `LLM_TEMPERATURE` | `0.7` | Sampling temperature |
| **TTS** | | |
| `HF_TOKEN` | *(none)* | Optional Hugging Face token for model download auth/rate limits |
| `TTS_MODEL_DIR` | *(none)* | Local pre-downloaded Chatterbox model directory |
| `TTS_DEVICE` | `auto` | `auto`, `cuda`, or `cpu` |
| `TTS_VOICE_PROMPT` | *(none)* | Path to voice cloning WAV (>5s) |
| **VAD** | | |
| `VAD_SPEECH_THRESHOLD` | `0.5` | Silero speech probability threshold |
| `VAD_SILENCE_THRESHOLD_MS` | `400` | Silence duration to end utterance |
| `VAD_MIN_SPEECH_MS` | `250` | Min speech duration to be real |
| **Other** | | |
| `OWNER_NAME` | *(none)* | Your name (used in greetings) |
| `RECORDINGS_DIR` | `recordings` | Where call WAVs are saved |
| `METADATA_DIR` | `metadata` | Where call metadata JSON is saved |
| `NTFY_TOPIC` | *(none)* | ntfy.sh topic for notifications |
| `NTFY_TOKEN` | *(none)* | Bearer token for authenticated ntfy topics |

</details>

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'chatterbox.tts_turbo'` | `chatterbox-tts` is too old. Run: `pip install --no-deps "chatterbox-tts>=0.1.5"` |
| `Permission denied` under `venv/.../site-packages` | You likely ran setup with `sudo`. Fix with `sudo rm -rf venv` then rerun `./setup.sh` as your normal user. |
| Demo mic doesn't work / no audio | Browsers require `localhost` or HTTPS for mic access. Use `http://localhost:8080/demo` (not a LAN IP). For remote access, use `python main.py --demo --share`. |
| PyTorch CUDA not available after install | If you're on macOS/CPU this is expected: keep `STT_DEVICE=auto` and `TTS_DEVICE=auto`. If you're on NVIDIA, reinstall CUDA PyTorch: `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124` |
| Chatterbox model download fails without auth | Prefer shipping a local bundle (`TTS_MODEL_DIR` or `models/chatterbox`). Otherwise set `HF_TOKEN` in `.env` or run `hf auth login`, then restart. |
| Slow responses in CPU mode | Set `STT_MODEL=base` and `STT_COMPUTE_TYPE=int8`, and use a smaller local LLM model. |
| `CUDA out of memory` | Use a smaller STT model (`STT_MODEL=base`) or lower precision (`STT_COMPUTE_TYPE=int8`). |
| Connection refused on port 1234/11434 | Start your LLM server first (`LM Studio` on 1234 or `Ollama` on 11434), then run `python main.py`. |

## Acknowledgments

- [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) by Resemble AI -- voice cloning
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) -- CTranslate2 Whisper implementation
- [Silero VAD](https://github.com/snakers4/silero-vad) -- voice activity detection
- [Gradio](https://gradio.app) -- demo UI and HTTPS tunneling
- [SignalWire](https://signalwire.com) -- telephony
- [LM Studio](https://lmstudio.ai) -- local LLM server
- [Ollama](https://ollama.com) -- local LLM runtime
- [ntfy](https://ntfy.sh) -- push notifications

## License

[MIT](LICENSE)
