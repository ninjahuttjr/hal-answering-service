"""Gradio-based demo UI for HAL Answering Service."""

import json
import logging

import gradio as gr

from config import Config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Check title mapping (friendly names for readiness check IDs)
# ---------------------------------------------------------------------------
CHECK_TITLES = {
    "llm_reachable": "LLM Endpoint",
    "llm_model": "Model",
    "voice_prompt": "Voice Prompt",
    "tts_model_bundle": "TTS Bundle",
    "sessions": "Demo Sessions",
    "capacity": "Call Capacity",
}


def _render_checks_html(checks: list[dict]) -> str:
    """Render readiness checks as styled HTML."""
    if not checks:
        return '<p style="color:#3a2828;font-size:0.76rem;">No checks available.</p>'

    rows = []
    for check in checks:
        failed = not check.get("ok", False)
        severity = check.get("severity", "error")
        if failed:
            level = "warn" if severity == "warning" else "error"
        else:
            level = "ok"
        title = CHECK_TITLES.get(check.get("id", ""), check.get("id", "")).replace("_", " ")
        message = check.get("message", "")

        if level == "ok":
            dot_color = "#3db86a"
            dot_shadow = "rgba(61,184,106,0.15)"
        elif level == "warn":
            dot_color = "#d4a32c"
            dot_shadow = "rgba(212,163,44,0.15)"
        else:
            dot_color = "#e05555"
            dot_shadow = "rgba(224,85,85,0.15)"

        rows.append(
            f'<div style="display:flex;align-items:flex-start;gap:10px;padding:7px 0;'
            f'border-bottom:1px solid #1a1212;font-size:0.74rem;line-height:1.4;color:#a08888;">'
            f'<span style="width:7px;height:7px;border-radius:50%;margin-top:5px;flex-shrink:0;'
            f'background:{dot_color};box-shadow:0 0 6px {dot_shadow};"></span>'
            f'<div><span style="font-size:0.68rem;letter-spacing:0.08em;text-transform:uppercase;'
            f'color:#8a6a6a;margin-right:6px;font-weight:500;">{title}</span>'
            f'<span style="color:#6a5555;">{message}</span></div>'
            f'</div>'
        )

    return f'<div style="padding:4px 8px;">{"".join(rows)}</div>'


def _render_status_summary(checks: list[dict]) -> tuple[str, str]:
    """Return a status summary (tone, text)."""
    blocking = sum(1 for c in checks if not c.get("ok") and c.get("severity") != "warning")
    warnings = sum(1 for c in checks if not c.get("ok") and c.get("severity") == "warning")
    if blocking > 0:
        return "error", "Blocked: fix readiness errors before starting."
    elif warnings > 0:
        return "warn", "Ready with warnings."
    return "ok", "Ready. All checks healthy."


# ---------------------------------------------------------------------------
# The full page HTML — CSS, layout, and JS all in one gr.HTML component.
# This avoids any Gradio CSS/JS injection issues.
# ---------------------------------------------------------------------------
FULL_PAGE_HTML = """
<div id="hal-hero">
  <div id="hal-eye-outer">
    <div id="hal-eye-shell">
      <div id="hal-eye" class="off"></div>
    </div>
  </div>
  <h1 id="hal-title">HAL 9000</h1>
  <p id="hal-subtitle">Answering Service</p>
</div>

<div id="hal-call-bar">
  <span id="hal-timer">00:00</span>
  <span id="hal-status" class="warn">Initializing...</span>
</div>

<div id="hal-level-wrap">
  <span id="hal-level-label">MIC</span>
  <div id="hal-level-track">
    <div id="hal-level-fill"></div>
  </div>
</div>

<div id="hal-transcript">
  <div class="hal-transcript-placeholder">
    <div class="hal-tp-icon">&#9678;</div>
    <div class="hal-tp-text">Press Start to begin a call with HAL</div>
  </div>
</div>
"""

# ---------------------------------------------------------------------------
# Standalone JS — injected via mount_gradio_app(head=<script>...)
# Runs once on page load as a self-executing function.
# ---------------------------------------------------------------------------
HAL_JS = """
() => {
  // ── Mu-law codec (ITU-T G.711) ──
  const MULAW_BIAS = 0x84;
  const MULAW_CLIP = 32635;
  const MULAW_EXP_TABLE = [0, 132, 396, 924, 1980, 4092, 8316, 16764];

  function linearToMulaw(sample) {
    sample = Math.max(-32768, Math.min(32767, Math.round(sample)));
    const sign = sample < 0 ? 0x80 : 0;
    if (sample < 0) sample = -sample;
    sample = Math.min(sample, MULAW_CLIP);
    sample += MULAW_BIAS;
    let exponent = 7;
    for (let expMask = 0x4000; exponent > 0; exponent--, expMask >>= 1) {
      if (sample >= expMask) break;
    }
    const mantissa = (sample >> (exponent + 3)) & 0x0f;
    return ~(sign | (exponent << 4) | mantissa) & 0xff;
  }

  function mulawToLinear(mulaw) {
    mulaw = ~mulaw & 0xff;
    const sign = mulaw & 0x80;
    const exponent = (mulaw >> 4) & 0x07;
    const mantissa = mulaw & 0x0f;
    let sample = MULAW_EXP_TABLE[exponent] + (mantissa << (exponent + 3));
    if (sign) sample = -sample;
    return sample;
  }

  // ── Resampling ──
  function downsample(float32Buf, fromRate, toRate) {
    const ratio = fromRate / toRate;
    const outLen = Math.floor(float32Buf.length / ratio);
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const srcIdx = i * ratio;
      const floor_ = Math.floor(srcIdx);
      const ceil_ = Math.min(floor_ + 1, float32Buf.length - 1);
      const frac = srcIdx - floor_;
      out[i] = float32Buf[floor_] * (1 - frac) + float32Buf[ceil_] * frac;
    }
    return out;
  }

  function upsample(int16Samples, fromRate, toRate) {
    const ratio = toRate / fromRate;
    const outLen = Math.round(int16Samples.length * ratio);
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const srcIdx = i / ratio;
      const floor_ = Math.floor(srcIdx);
      const ceil_ = Math.min(floor_ + 1, int16Samples.length - 1);
      const frac = srcIdx - floor_;
      const sample = int16Samples[floor_] * (1 - frac) + int16Samples[ceil_] * frac;
      out[i] = sample / 32768;
    }
    return out;
  }

  // ── HalAudio namespace ──
  const TARGET_RATE = 8000;
  const FRAME_SIZE = 160;

  window.HalAudio = {
    ws: null,
    audioCtx: null,
    micStream: null,
    scriptNode: null,
    micSource: null,
    isCallActive: false,
    micBuffer: new Float32Array(0),
    playbackTime: 0,
    activeSources: [],
    isPlaying: false,
    callTimerHandle: null,
    callStartedAtMs: 0,

    setEye(state) {
      const eye = document.getElementById("hal-eye");
      if (eye) eye.className = state;
    },

    setStatus(text, tone) {
      const el = document.getElementById("hal-status");
      if (el) {
        el.textContent = text;
        el.className = tone || "warn";
      }
    },

    setTimer(text) {
      const el = document.getElementById("hal-timer");
      if (el) el.textContent = text;
    },

    setMicLevel(pct) {
      const bar = document.getElementById("hal-level-fill");
      if (bar) bar.style.width = pct + "%";
      const wrap = document.getElementById("hal-level-wrap");
      if (wrap) wrap.style.display = pct >= 0 ? "flex" : "none";
    },

    appendTranscript(role, text) {
      const box = document.getElementById("hal-transcript");
      if (!box) return;
      const ph = box.querySelector(".hal-transcript-placeholder");
      if (ph) ph.remove();

      const entry = document.createElement("div");
      entry.className = "hal-transcript-entry";
      const roleSpan = document.createElement("div");
      roleSpan.className = "hal-role " + (role === "agent" ? "hal-role-agent" : "hal-role-caller");
      roleSpan.textContent = role === "agent" ? "HAL" : "YOU";
      const textSpan = document.createElement("div");
      textSpan.className = "hal-text";
      textSpan.textContent = text;
      entry.appendChild(roleSpan);
      entry.appendChild(textSpan);
      box.appendChild(entry);
      box.scrollTop = box.scrollHeight;
    },

    startTimer() {
      this.stopTimer();
      this.callStartedAtMs = Date.now();
      this.setTimer("00:00");
      this.callTimerHandle = setInterval(() => {
        const ms = Date.now() - this.callStartedAtMs;
        const totalSec = Math.max(0, Math.floor(ms / 1000));
        const min = String(Math.floor(totalSec / 60)).padStart(2, "0");
        const sec = String(totalSec % 60).padStart(2, "0");
        this.setTimer(min + ":" + sec);
      }, 250);
    },

    stopTimer() {
      if (this.callTimerHandle) {
        clearInterval(this.callTimerHandle);
        this.callTimerHandle = null;
      }
      this.setTimer("00:00");
    },

    queuePlayback(mulawBase64) {
      if (!this.audioCtx) return;
      const binary = atob(mulawBase64);
      const mulawBytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) {
        mulawBytes[i] = binary.charCodeAt(i);
      }
      const int16Samples = new Int16Array(mulawBytes.length);
      for (let i = 0; i < mulawBytes.length; i++) {
        int16Samples[i] = mulawToLinear(mulawBytes[i]);
      }
      const float32 = upsample(int16Samples, TARGET_RATE, this.audioCtx.sampleRate);
      const buffer = this.audioCtx.createBuffer(1, float32.length, this.audioCtx.sampleRate);
      buffer.getChannelData(0).set(float32);
      const source = this.audioCtx.createBufferSource();
      source.buffer = buffer;
      source.connect(this.audioCtx.destination);
      const now = this.audioCtx.currentTime;
      const startAt = Math.max(now + 0.02, this.playbackTime);
      source.start(startAt);
      this.playbackTime = startAt + buffer.duration;
      this.isPlaying = true;
      const entry = { source };
      this.activeSources.push(entry);
      source.onended = () => {
        const idx = this.activeSources.indexOf(entry);
        if (idx !== -1) this.activeSources.splice(idx, 1);
        if (this.activeSources.length === 0) {
          this.isPlaying = false;
          if (this.isCallActive) {
            this.setEye("listening");
            this.setStatus("Listening...", "ok");
          }
        }
      };
    },

    clearPlayback() {
      for (const s of this.activeSources) {
        try { s.source.stop(); } catch (_) {}
      }
      this.activeSources = [];
      this.playbackTime = 0;
      this.isPlaying = false;
    },

    onAudioProcess(e) {
      if (!this.isCallActive || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;
      const input = e.inputBuffer.getChannelData(0);
      let sum = 0;
      for (let i = 0; i < input.length; i++) sum += input[i] * input[i];
      const rms = Math.sqrt(sum / Math.max(1, input.length));
      const pct = Math.min(100, rms * 500);
      this.setMicLevel(pct);

      const downsampled = downsample(input, this.audioCtx.sampleRate, TARGET_RATE);
      const newBuf = new Float32Array(this.micBuffer.length + downsampled.length);
      newBuf.set(this.micBuffer);
      newBuf.set(downsampled, this.micBuffer.length);
      this.micBuffer = newBuf;

      while (this.micBuffer.length >= FRAME_SIZE) {
        const frame = this.micBuffer.slice(0, FRAME_SIZE);
        this.micBuffer = this.micBuffer.slice(FRAME_SIZE);
        const mulawBytes = new Uint8Array(FRAME_SIZE);
        for (let i = 0; i < FRAME_SIZE; i++) {
          mulawBytes[i] = linearToMulaw(frame[i] * 32768);
        }
        let binary = "";
        for (let i = 0; i < mulawBytes.length; i++) {
          binary += String.fromCharCode(mulawBytes[i]);
        }
        this.ws.send(JSON.stringify({ event: "media", media: { payload: btoa(binary) } }));
      }
    },

    async startCall() {
      if (this.isCallActive) return;

      // Clear transcript
      const box = document.getElementById("hal-transcript");
      if (box) box.innerHTML = "";

      this.setStatus("Requesting microphone access...", "warn");

      // Browsers kill navigator.mediaDevices on non-localhost HTTP.
      // Auto-redirect to localhost if on LAN IP.
      if (!navigator.mediaDevices) {
        const port = location.port || "8080";
        const localhostUrl = "http://localhost:" + port + location.pathname;
        this.setStatus("Redirecting to localhost for mic access...", "warn");
        window.location.href = localhostUrl;
        return;
      }

      try {
        this.micStream = await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
        });
      } catch (err) {
        const msg = (err.name === "NotAllowedError")
          ? "Microphone denied. Allow mic and try again."
          : (err.name === "NotFoundError")
            ? "No microphone found."
            : "Mic error: " + err.message;
        this.setStatus(msg, "error");
        return;
      }

      this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      this.micSource = this.audioCtx.createMediaStreamSource(this.micStream);
      this.scriptNode = this.audioCtx.createScriptProcessor(4096, 1, 1);
      this.scriptNode.onaudioprocess = (e) => this.onAudioProcess(e);
      this.micSource.connect(this.scriptNode);
      this.scriptNode.connect(this.audioCtx.destination);

      this.micBuffer = new Float32Array(0);
      this.playbackTime = 0;
      this.activeSources = [];
      this.isPlaying = false;
      this.isCallActive = true;

      this.setMicLevel(0);
      this.setStatus("Connecting...", "warn");
      this.setEye("listening");

      const protocol = location.protocol === "https:" ? "wss:" : "ws:";
      this.ws = new WebSocket(protocol + "//" + location.host + "/demo-stream");

      this.ws.onopen = () => {
        this.startTimer();
        this.setStatus("Connected. HAL is greeting you...", "ok");
        this.setEye("speaking");
      };

      this.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.event === "media") {
          this.setEye("speaking");
          this.setStatus("HAL is speaking...", "ok");
          this.queuePlayback(data.media.payload);
        } else if (data.event === "clear") {
          this.clearPlayback();
        } else if (data.event === "transcript") {
          this.appendTranscript(data.role, data.text);
        } else if (data.event === "hangup") {
          this.setStatus("HAL ended the call.", "warn");
          this.setEye("off");
          this.cleanup();
          return;
        }
      };

      this.ws.onclose = (event) => {
        if (this.isCallActive) {
          const reason = event.reason ? " (" + event.reason + ")" : "";
          this.setStatus("Call ended" + reason + ".", "warn");
          this.setEye("off");
          this.cleanup();
        }
      };

      this.ws.onerror = () => {
        this.setStatus("WebSocket error. Is server running?", "error");
        this.setEye("off");
        this.cleanup();
      };
    },

    endCall() {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ event: "stop" }));
      }
      this.setStatus("Ending call...", "warn");
      this.setEye("off");
      this.cleanup();
    },

    cleanup() {
      this.isCallActive = false;
      this.stopTimer();
      if (this.scriptNode) { this.scriptNode.disconnect(); this.scriptNode = null; }
      if (this.micSource) { this.micSource.disconnect(); this.micSource = null; }
      if (this.micStream) { this.micStream.getTracks().forEach(t => t.stop()); this.micStream = null; }
      this.clearPlayback();
      if (this.ws) { try { this.ws.close(); } catch (_) {} this.ws = null; }
      if (this.audioCtx) { try { this.audioCtx.close(); } catch (_) {} this.audioCtx = null; }
      this.setMicLevel(-1);
    }
  };

  // Initialize eye state
  window.HalAudio.setEye("off");

  // Watch for status signals from Gradio callbacks.
  // Gradio strips <script> tags from gr.HTML output, so the Python callback
  // emits a hidden <div id="hal-status-signal" data-tone="..." data-text="...">
  // and we pick it up with a MutationObserver.
  const observer = new MutationObserver(() => {
    const signal = document.getElementById("hal-status-signal");
    if (signal && !window.HalAudio.isCallActive) {
      const tone = signal.getAttribute("data-tone");
      const text = signal.getAttribute("data-text");
      if (tone && text) {
        window.HalAudio.setStatus(text, tone);
      }
      signal.remove();
    }
  });
  observer.observe(document.body, { childList: true, subtree: true });

  console.log("[HAL] Audio pipeline initialized");
}
"""

HAL_CSS = r"""
/* ═══════════════════════════════════════════════════════════
   HAL 9000 — Professional Dark UI
   ═══════════════════════════════════════════════════════════ */

/* ── Foundation ── */
:root {
  --h-bg: #080606;
  --h-surface: #0e0b0b;
  --h-elevated: #141010;
  --h-border: #221818;
  --h-border-subtle: #1a1212;
  --h-text: #d8c8c8;
  --h-text-dim: #7a6262;
  --h-text-bright: #f0e0e0;
  --h-red: #bf1d1d;
  --h-red-soft: rgba(191, 29, 29, 0.35);
  --h-red-glow: rgba(191, 29, 29, 0.12);
  --h-green: #3db86a;
  --h-yellow: #d4a32c;
  --h-err: #e05555;
  --h-font: "SF Mono", "Fira Code", "Cascadia Code", "JetBrains Mono", "Courier New", monospace;
  --h-radius: 12px;
}

/* ── Global reset ── */
*, *::before, *::after { box-sizing: border-box; }

body, html {
  background: var(--h-bg) !important;
  color: var(--h-text) !important;
  font-family: var(--h-font) !important;
}

/* ── Gradio nuclear reset ── */
.gradio-container,
.gradio-container .main,
.gradio-container .fillable,
gradio-app {
  background: var(--h-bg) !important;
  color: var(--h-text) !important;
  font-family: var(--h-font) !important;
  max-width: 720px !important;
  margin: 0 auto !important;
  padding-top: 0 !important;
}

/* Kill every Gradio wrapper border/bg/shadow */
.gradio-container .block,
.gradio-container .form,
.gradio-container .panel,
.gradio-container .contain,
.gradio-container .wrap,
.gradio-container .html-container,
.gradio-container .hide-container,
.gradio-container .column,
.gradio-container .padded,
.gradio-container [class*="svelte-"] {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  outline: none !important;
}

/* Restore borders only on actual interactive wrappers inside accordions */
.gradio-container details .wrap {
  background: var(--h-surface) !important;
  border: 1px solid var(--h-border) !important;
  border-radius: 8px !important;
}

/* Kill all Gradio footer/chrome */
footer, .gradio-container footer, .built-with,
div[class*="footer"], .api-link, [class*="api-link"],
.settings-icon, .toast-wrap { display: none !important; }

/* ── Background atmosphere ── */
body::before {
  content: "";
  position: fixed;
  inset: 0;
  background:
    radial-gradient(ellipse 800px 400px at 50% -100px, rgba(191, 29, 29, 0.08), transparent 70%),
    radial-gradient(ellipse 600px 300px at 50% 100%, rgba(191, 29, 29, 0.03), transparent 60%);
  pointer-events: none;
  z-index: 0;
}

/* ── HAL Eye ── */
#hal-hero {
  text-align: center;
  padding: 40px 16px 8px;
  position: relative;
}

#hal-eye-outer {
  width: 160px;
  height: 160px;
  margin: 0 auto 18px;
  position: relative;
  display: grid;
  place-items: center;
}

/* Subtle ring halo */
#hal-eye-outer::before {
  content: "";
  position: absolute;
  inset: -8px;
  border-radius: 50%;
  border: 1px solid rgba(191, 29, 29, 0.08);
}

#hal-eye-shell {
  width: 148px;
  height: 148px;
  border-radius: 50%;
  display: grid;
  place-items: center;
  border: 1.5px solid #2a1616;
  background:
    radial-gradient(circle, rgba(191, 29, 29, 0.06) 0%, rgba(14, 10, 10, 0.98) 55%),
    linear-gradient(180deg, #151010, #0a0808);
  box-shadow:
    inset 0 0 30px rgba(191, 29, 29, 0.06),
    0 0 60px rgba(191, 29, 29, 0.03),
    0 0 0 4px #0c0808,
    0 0 0 5.5px #201414;
}

#hal-eye {
  width: 110px;
  height: 110px;
  border-radius: 50%;
  background:
    radial-gradient(circle at 36% 30%,
      #ffdada 0%, #ff7070 14%, #d02020 34%, #801010 62%, #280606 100%);
  box-shadow: 0 0 50px var(--h-red-soft), 0 0 100px var(--h-red-glow);
  transition: all 300ms cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
  overflow: hidden;
}

/* Glass highlight */
#hal-eye::before {
  content: "";
  position: absolute;
  top: 12%;
  left: 20%;
  width: 30%;
  height: 30%;
  background: radial-gradient(circle, rgba(255,255,255,0.15), transparent 70%);
  border-radius: 50%;
  pointer-events: none;
}

/* Lens flare streak */
#hal-eye::after {
  content: "";
  position: absolute;
  top: 24%;
  left: 10%;
  width: 55%;
  height: 8%;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
  border-radius: 50%;
  transform: rotate(-18deg);
  pointer-events: none;
}

#hal-eye.off {
  background: radial-gradient(circle at 36% 30%,
    #606060 0%, #484848 25%, #2a2a2a 55%, #141414 100%);
  box-shadow: 0 0 15px rgba(100, 100, 100, 0.05);
}
#hal-eye.off::before, #hal-eye.off::after { opacity: 0.15; }

#hal-eye.listening {
  box-shadow: 0 0 50px rgba(191, 29, 29, 0.5), 0 0 100px rgba(191, 29, 29, 0.15);
  transform: scale(1.015);
  animation: hal-breathe 3s ease-in-out infinite;
}

#hal-eye.speaking {
  box-shadow: 0 0 60px rgba(191, 29, 29, 0.7), 0 0 120px rgba(191, 29, 29, 0.25);
  animation: hal-pulse 1.2s ease-in-out infinite;
}

@keyframes hal-pulse {
  0%, 100% { box-shadow: 0 0 55px rgba(191, 29, 29, 0.6), 0 0 110px rgba(191, 29, 29, 0.2); transform: scale(1.01); }
  50%      { box-shadow: 0 0 80px rgba(191, 29, 29, 0.9), 0 0 150px rgba(191, 29, 29, 0.35); transform: scale(1.025); }
}

@keyframes hal-breathe {
  0%, 100% { box-shadow: 0 0 45px rgba(191, 29, 29, 0.4), 0 0 90px rgba(191, 29, 29, 0.1); transform: scale(1.015); }
  50%      { box-shadow: 0 0 60px rgba(191, 29, 29, 0.55), 0 0 120px rgba(191, 29, 29, 0.18); transform: scale(1.03); }
}

#hal-title {
  font-size: 1.1rem;
  letter-spacing: 0.22em;
  color: var(--h-text-bright);
  margin: 0;
  font-family: var(--h-font);
  text-transform: uppercase;
  font-weight: 300;
}

#hal-subtitle {
  color: var(--h-text-dim);
  font-size: 0.72rem;
  margin-top: 3px;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  font-weight: 400;
}

/* ── Call Status Bar ── */
#hal-call-bar {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 14px;
  padding: 10px 0 14px;
}

#hal-timer {
  color: var(--h-text-dim);
  font-family: var(--h-font);
  font-size: 0.78rem;
  font-variant-numeric: tabular-nums;
  letter-spacing: 0.05em;
  opacity: 0.7;
}

#hal-status {
  font-family: var(--h-font);
  font-size: 0.78rem;
  color: var(--h-text-dim);
  letter-spacing: 0.02em;
}

#hal-status.ok { color: var(--h-green); }
#hal-status.warn { color: var(--h-yellow); }
#hal-status.error { color: var(--h-err); }

/* ── Mic Level ── */
#hal-level-wrap {
  display: none;
  align-items: center;
  gap: 10px;
  padding: 4px 0 8px;
  margin: 0 auto;
  max-width: 320px;
}

#hal-level-label {
  font-size: 0.62rem;
  color: var(--h-text-dim);
  font-family: var(--h-font);
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

#hal-level-track {
  flex: 1;
  height: 3px;
  border-radius: 2px;
  background: var(--h-border);
  overflow: hidden;
}

#hal-level-fill {
  height: 100%;
  width: 0%;
  background: linear-gradient(90deg, #6a1a1a, var(--h-red));
  border-radius: 2px;
  transition: width 60ms linear;
}

/* ── Transcript ── */
#hal-transcript {
  border: 1px solid var(--h-border-subtle);
  border-radius: var(--h-radius);
  background: var(--h-surface);
  min-height: 220px;
  max-height: 380px;
  overflow-y: auto;
  padding: 16px 18px;
  font-family: var(--h-font);
  margin-bottom: 4px;
}

.hal-transcript-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 180px;
  gap: 8px;
}

.hal-tp-icon {
  font-size: 1.8rem;
  color: #2a1818;
  line-height: 1;
}

.hal-tp-text {
  color: #3a2828;
  font-size: 0.76rem;
  letter-spacing: 0.04em;
}

.hal-transcript-entry {
  margin-bottom: 12px;
  line-height: 1.45;
}

.hal-transcript-entry:last-child { margin-bottom: 0; }

.hal-role {
  font-size: 0.65rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  font-weight: 600;
  opacity: 0.85;
}

.hal-role-agent { color: #c45050; }
.hal-role-caller { color: #8a7070; }

.hal-text {
  margin-top: 3px;
  font-size: 0.8rem;
  color: var(--h-text);
  line-height: 1.5;
}

#hal-transcript::-webkit-scrollbar { width: 4px; }
#hal-transcript::-webkit-scrollbar-track { background: transparent; }
#hal-transcript::-webkit-scrollbar-thumb { background: #221818; border-radius: 2px; }
#hal-transcript::-webkit-scrollbar-thumb:hover { background: #3a2828; }

/* ── Buttons ── */
.gradio-container button.primary {
  background: linear-gradient(180deg, #5a1616 0%, #3d0e0e 100%) !important;
  border: 1px solid rgba(191, 29, 29, 0.4) !important;
  color: #f0d0d0 !important;
  font-family: var(--h-font) !important;
  border-radius: 10px !important;
  font-weight: 400 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.1em !important;
  font-size: 0.78rem !important;
  padding: 12px 24px !important;
  transition: all 200ms ease !important;
  position: relative !important;
  overflow: hidden !important;
}

.gradio-container button.primary:hover {
  border-color: rgba(191, 29, 29, 0.6) !important;
  box-shadow: 0 0 20px rgba(191, 29, 29, 0.15), inset 0 1px 0 rgba(255,255,255,0.03) !important;
  background: linear-gradient(180deg, #6a1c1c 0%, #451111 100%) !important;
}

.gradio-container button.primary:active {
  transform: scale(0.98) !important;
}

.gradio-container button.stop {
  background: transparent !important;
  border: 1px solid #3a2222 !important;
  color: #a07070 !important;
  font-family: var(--h-font) !important;
  border-radius: 10px !important;
  font-size: 0.76rem !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
  padding: 12px 24px !important;
  transition: all 200ms ease !important;
}

.gradio-container button.stop:hover {
  border-color: #5a2828 !important;
  color: #d08080 !important;
  background: rgba(191, 29, 29, 0.06) !important;
}

.gradio-container button.secondary {
  background: var(--h-surface) !important;
  border: 1px solid var(--h-border) !important;
  color: var(--h-text-dim) !important;
  font-family: var(--h-font) !important;
  border-radius: 10px !important;
  font-size: 0.74rem !important;
  letter-spacing: 0.05em !important;
  transition: all 200ms ease !important;
}

.gradio-container button.secondary:hover {
  border-color: #3a2828 !important;
  color: var(--h-text) !important;
  background: var(--h-elevated) !important;
}

/* ── Accordions ── */
.gradio-container details {
  background: var(--h-surface) !important;
  border: 1px solid var(--h-border-subtle) !important;
  border-radius: var(--h-radius) !important;
  overflow: hidden !important;
  margin-top: 8px !important;
}

.gradio-container details summary {
  color: var(--h-text-dim) !important;
  font-family: var(--h-font) !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  padding: 12px 16px !important;
  cursor: pointer !important;
  background: var(--h-surface) !important;
  transition: color 150ms ease !important;
}

.gradio-container details summary:hover {
  color: var(--h-text) !important;
}

.gradio-container details[open] summary {
  border-bottom: 1px solid var(--h-border-subtle) !important;
}

/* ── Inputs inside accordions ── */
.gradio-container details input[type="text"],
.gradio-container details input[type="password"],
.gradio-container details textarea,
.gradio-container details select {
  background: var(--h-bg) !important;
  border: 1px solid var(--h-border) !important;
  border-radius: 8px !important;
  color: var(--h-text) !important;
  font-family: var(--h-font) !important;
  font-size: 0.78rem !important;
  padding: 8px 12px !important;
  transition: border-color 150ms ease !important;
}

.gradio-container details input:focus,
.gradio-container details select:focus {
  border-color: rgba(191, 29, 29, 0.3) !important;
  outline: none !important;
  box-shadow: 0 0 0 3px rgba(191, 29, 29, 0.06) !important;
}

/* Restore accordion inner wrap borders */
.gradio-container details .wrap {
  background: var(--h-surface) !important;
  border: 1px solid var(--h-border) !important;
  border-radius: 8px !important;
}

.gradio-container label,
.gradio-container .label-text {
  color: var(--h-text-dim) !important;
  font-family: var(--h-font) !important;
  font-size: 0.68rem !important;
  letter-spacing: 0.05em !important;
  text-transform: uppercase !important;
}

/* ── Dropdowns ── */
.gradio-container ul[role="listbox"] {
  background: var(--h-elevated) !important;
  border: 1px solid var(--h-border) !important;
  border-radius: 8px !important;
  overflow: hidden !important;
}

.gradio-container ul[role="listbox"] li {
  color: var(--h-text) !important;
  font-family: var(--h-font) !important;
  font-size: 0.78rem !important;
}

.gradio-container ul[role="listbox"] li:hover,
.gradio-container ul[role="listbox"] li[aria-selected="true"] {
  background: rgba(191, 29, 29, 0.08) !important;
}

/* ── Row spacing ── */
.gradio-container .row { gap: 8px !important; }

/* ── Toast notifications ── */
.toast-wrap { display: none !important; }
"""

# ---------------------------------------------------------------------------
# Gradio app factory
# ---------------------------------------------------------------------------
def create_gradio_app(
    config: Config,
    status_checker=None,
    config_reader=None,
    config_writer=None,
):
    """Create the Gradio demo UI.

    Args:
        config: The application Config object.
        status_checker: Async callable that returns the status dict (same as /demo/status).
        config_reader: Async callable that returns the config dict (same as GET /demo/config).
        config_writer: Async callable(payload) that applies/saves config (same as POST /demo/config).
    """

    with gr.Blocks(title="HAL Answering Service") as demo:

        # ── Full page HTML: eye, status, transcript — all in one component ──
        gr.HTML(FULL_PAGE_HTML)

        # ── Call Controls ──
        with gr.Row():
            start_btn = gr.Button("Start Demo Call", variant="primary", scale=2)
            end_btn = gr.Button("End Call", variant="stop", scale=1)

        # ── Button JS handlers ──
        start_btn.click(
            fn=None,
            inputs=None,
            outputs=None,
            js="() => { if(window.HalAudio) window.HalAudio.startCall(); else console.error('[HAL] HalAudio not initialized'); }",
        )

        end_btn.click(
            fn=None,
            inputs=None,
            outputs=None,
            js="() => { if(window.HalAudio) window.HalAudio.endCall(); }",
        )

        # ── Readiness Checks ──
        with gr.Accordion("Readiness Checks", open=False):
            checks_html = gr.HTML(value='<p style="color:#3a2828;font-size:0.74rem;padding:8px;">Loading...</p>')
            refresh_btn = gr.Button("Refresh Checks", variant="secondary")

        # ── Settings ──
        with gr.Accordion("Settings", open=False):
            with gr.Row():
                llm_provider = gr.Dropdown(
                    choices=["auto", "lmstudio", "ollama", "openai_compatible"],
                    value=config.llm_provider or "auto",
                    label="LLM Provider",
                )
                llm_url = gr.Textbox(
                    value=config.llm_base_url or "",
                    label="LLM Base URL",
                )
            with gr.Row():
                llm_model = gr.Textbox(
                    value=config.llm_model or "",
                    label="LLM Model",
                )
                llm_api_key = gr.Textbox(
                    value=config.llm_api_key or "",
                    label="LLM API Key",
                    type="password",
                )
            with gr.Row():
                stt_model_dd = gr.Dropdown(
                    choices=["large-v3-turbo", "large-v3", "medium", "small", "base", "tiny"],
                    value=config.stt_model or "large-v3-turbo",
                    label="STT Model",
                )
                stt_language = gr.Textbox(
                    value=config.stt_language or "en",
                    label="STT Language",
                )
                owner_name = gr.Textbox(
                    value=config.owner_name or "",
                    label="Owner Name",
                )
            with gr.Row():
                apply_btn = gr.Button("Apply Runtime", variant="secondary")
                save_btn = gr.Button("Save to .env", variant="primary")
            settings_status = gr.HTML(
                value='<p style="color:#3a2828;font-size:0.7rem;letter-spacing:0.03em;">Changes apply to new calls.</p>'
            )

        # ── Readiness check callback ──
        async def do_check_status():
            if status_checker is None:
                return '<p style="color:#e05555;font-size:0.74rem;">Status checker not available.</p>'
            try:
                data = await status_checker()
            except Exception as e:
                return f'<p style="color:#e05555;font-size:0.74rem;">Status check failed: {e}</p>'

            checks = data.get("checks", [])
            tone, summary_text = _render_status_summary(checks)

            # Return the checks HTML — status bar is updated via JS separately
            checks_out = _render_checks_html(checks)

            # We can't use <script> tags in gr.HTML output (Gradio strips them).
            # Instead, we include a hidden element with the status data that
            # a MutationObserver in the main JS picks up.
            status_data = json.dumps({"tone": tone, "text": summary_text})
            status_signal = (
                f'<div id="hal-status-signal" style="display:none" '
                f'data-tone="{tone}" data-text="{summary_text}"></div>'
            )
            return checks_out + status_signal

        refresh_btn.click(fn=do_check_status, inputs=None, outputs=[checks_html])
        demo.load(fn=do_check_status, inputs=None, outputs=[checks_html])

        # ── Settings handlers ──
        def _settings_msg(msg, is_warning=False):
            color = "#d4a32c" if is_warning else "#3db86a"
            return f'<p style="color:{color};font-size:0.72rem;letter-spacing:0.02em;">{msg}</p>'

        def _settings_error(msg):
            return f'<p style="color:#e05555;font-size:0.72rem;">{msg}</p>'

        def _collect_values(provider, url, model, api_key, stt_m, stt_lang, owner):
            return {
                "LLM_PROVIDER": provider or "",
                "LLM_BASE_URL": url or "",
                "LLM_MODEL": model or "",
                "LLM_API_KEY": api_key or "",
                "STT_MODEL": stt_m or "",
                "STT_LANGUAGE": stt_lang or "",
                "OWNER_NAME": owner or "",
            }

        async def do_apply_settings(provider, url, model, api_key, stt_m, stt_lang, owner):
            if config_writer is None:
                return _settings_error("Config writer not available.")
            try:
                result = await config_writer({
                    "values": _collect_values(provider, url, model, api_key, stt_m, stt_lang, owner),
                    "apply_runtime": True,
                    "save": False,
                })
                if not isinstance(result, dict):
                    return _settings_error("Unexpected response from config writer.")
            except Exception as e:
                return _settings_error(f"Apply failed: {e}")

            applied = result.get("applied_runtime", [])
            restart = result.get("requires_restart", [])
            msg = f"Applied {len(applied)} runtime setting(s)."
            if restart:
                msg += f" {len(restart)} require restart ({', '.join(restart[:3])})."
            return _settings_msg(msg, bool(restart))

        async def do_save_settings(provider, url, model, api_key, stt_m, stt_lang, owner):
            if config_writer is None:
                return _settings_error("Config writer not available.")
            try:
                result = await config_writer({
                    "values": _collect_values(provider, url, model, api_key, stt_m, stt_lang, owner),
                    "apply_runtime": True,
                    "save": True,
                })
                if not isinstance(result, dict):
                    return _settings_error("Unexpected response from config writer.")
            except Exception as e:
                return _settings_error(f"Save failed: {e}")

            applied = result.get("applied_runtime", [])
            restart = result.get("requires_restart", [])
            msg = f"Saved to .env. Applied {len(applied)} runtime setting(s)."
            if restart:
                msg += f" {len(restart)} require restart."
            return _settings_msg(msg, bool(restart))

        settings_inputs = [llm_provider, llm_url, llm_model, llm_api_key, stt_model_dd, stt_language, owner_name]

        apply_btn.click(fn=do_apply_settings, inputs=settings_inputs, outputs=[settings_status])
        save_btn.click(fn=do_save_settings, inputs=settings_inputs, outputs=[settings_status])

    return demo
