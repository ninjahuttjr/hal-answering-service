# SuperCaller

AI-powered phone call screener that answers your phone with a HAL 9000 personality. Collects caller name and reason for calling, then sends you a summary notification.

Runs entirely on your own hardware — no cloud AI APIs required. Uses a local LLM, local speech-to-text, and local text-to-speech with voice cloning.

## How It Works

```
Incoming Call (SignalWire)
    |
    v
[WebSocket Media Stream]
    |
    v
Caller Audio --> VAD (Silero) --> STT (Faster-Whisper) --> LLM (local, OpenAI-compatible)
                                                              |
                                                              v
                                                     TTS (Chatterbox Turbo)
                                                              |
                                                              v
                                                     Audio back to caller
```

1. A call comes in through SignalWire, which forwards the audio as a real-time WebSocket stream.
2. **Silero VAD** detects when the caller starts and stops speaking.
3. **Faster-Whisper** transcribes the speech to text on GPU.
4. A **local LLM** (via LM Studio or any OpenAI-compatible server) generates a conversational response as HAL 9000.
5. **Chatterbox Turbo** synthesizes the response to speech with voice cloning.
6. The audio is encoded as G.711 mu-law and streamed back to the caller.
7. When the call ends, a summary is sent via [ntfy](https://ntfy.sh).

## Features

- **Real-time conversational AI** on phone calls with sub-second latency
- **HAL 9000 personality** — friendly, witty, and self-aware
- **Voice cloning** via Chatterbox Turbo (provide a 5s+ WAV sample)
- **Barge-in detection** — caller can interrupt the AI mid-sentence
- **Silence timeout** — prompts the caller if they go quiet, hangs up after repeated silence
- **Call recording** — saves mixed mono WAV files of both parties
- **Call summaries** via ntfy push notifications
- **Pre-recorded greetings** — instant playback, no TTS delay on pickup
- **Webhook signature validation** — rejects forged requests
- **WebSocket authentication** — validates CallSid from signed webhooks
- **Max concurrent calls** and **max call duration** limits
- **Prompt injection hardening** — input truncation, security instructions, XML escaping

## Requirements

- **Python 3.12+**
- **NVIDIA GPU** with CUDA support (for Whisper, Chatterbox Turbo, and Silero VAD)
- **SignalWire account** with a phone number configured to forward calls to your webhook
- **Local LLM server** — [LM Studio](https://lmstudio.ai/) or any OpenAI-compatible API
- **Public HTTPS endpoint** — Tailscale Funnel, Cloudflare Tunnel, ngrok, etc.

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/YOUR_USERNAME/SuperCaller.git
cd SuperCaller
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your values:

- **SignalWire credentials** — project ID, token, space name, phone number
- **PUBLIC_HOST** — your public hostname (e.g. from Tailscale Funnel or Cloudflare Tunnel)
- **HF_TOKEN** — Hugging Face token (required to download Chatterbox Turbo model on first run)
- **LLM_BASE_URL** — your local LLM server endpoint (default: `http://127.0.0.1:1234/v1` for LM Studio)
- **OWNER_NAME** — the name HAL will use when referring to you
- **TTS_VOICE_PROMPT** — (optional) path to a WAV file for voice cloning
- **NTFY_TOPIC** — (optional) ntfy.sh topic for call summary notifications

### 3. Start your local LLM

Start LM Studio (or any OpenAI-compatible server) with a model loaded. The default config expects it at `http://127.0.0.1:1234/v1`.

### 4. Expose your server publicly

You need a public HTTPS endpoint that forwards to port 8080. Examples:

```bash
# Tailscale Funnel
tailscale funnel 8080

# Cloudflare Tunnel
cloudflared tunnel --url http://localhost:8080

# ngrok
ngrok http 8080
```

Set `PUBLIC_HOST` in your `.env` to the resulting hostname.

### 5. Configure SignalWire

In your SignalWire dashboard, set your phone number's incoming call webhook to:

```
https://YOUR_PUBLIC_HOST/incoming-call
```

### 6. Run

```bash
python main.py
```

On first run, models will be downloaded automatically (Chatterbox Turbo, Faster-Whisper, Silero VAD). Subsequent starts are faster since models are cached.

## Architecture

| File | Purpose |
|------|---------|
| `main.py` | Entry point — loads models, pre-records greetings, starts server |
| `server.py` | FastAPI server — webhook handler, WebSocket media stream, ntfy notifications |
| `call_handler.py` | Per-call pipeline — VAD, STT, LLM, TTS, barge-in, silence detection, recording |
| `config.py` | Dataclass config loaded from environment variables |
| `llm.py` | OpenAI-compatible LLM client with streaming sentence extraction |
| `prompts.py` | HAL 9000 system prompt, greeting templates, summary prompt |
| `audio.py` | G.711 mu-law codec, resampling, Silero VAD wrapper |
| `stt.py` | Faster-Whisper speech-to-text wrapper |
| `tts.py` | Chatterbox Turbo text-to-speech wrapper |

## Voice Cloning

Chatterbox Turbo supports voice cloning from a reference WAV file. To use a custom voice:

1. Record a 5+ second WAV sample of the target voice
2. Set `TTS_VOICE_PROMPT=/path/to/sample.wav` in your `.env`

Two sample voice files are included: `hal9000.wav` and `eugene.wav`.

## License

[MIT](LICENSE)
