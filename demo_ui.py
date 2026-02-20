"""Native HTML/JS demo UI for HAL Answering Service."""

import json
import logging
from config import Config

log = logging.getLogger(__name__)

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
        return '<p class="hal-no-checks">No checks available.</p>'

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

        rows.append(
            f'<div class="hal-check-row">'
            f'<span class="hal-check-dot {level}"></span>'
            f'<div><span class="hal-check-title">{title}</span>'
            f'<span class="hal-check-msg">{message}</span></div>'
            f'</div>'
        )

    return f'<div class="hal-checks-container">{"".join(rows)}</div>'

def _render_status_summary(checks: list[dict]) -> tuple[str, str]:
    """Return a status summary (tone, text)."""
    blocking = sum(1 for c in checks if not c.get("ok") and c.get("severity") != "warning")
    warnings = sum(1 for c in checks if not c.get("ok") and c.get("severity") == "warning")
    if blocking > 0:
        return "error", "Blocked: fix readiness errors before starting."
    elif warnings > 0:
        return "warn", "Ready with warnings."
    return "ok", "Ready. All checks healthy."

def get_demo_html(status_data: dict, config: Config) -> str:
    checks = status_data.get("checks", [])
    tone, summary_text = _render_status_summary(checks)
    checks_html = _render_checks_html(checks)

    # Note: We put all the layout and styling in a standalone standard HTML skeleton.
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>HAL Answering Service</title>
  <style>
/* ═══════════════════════════════════════════════════════════
   HAL 9000 — Professional Dark UI (Native Base)
   ═══════════════════════════════════════════════════════════ */

/* ── Foundation ── */
:root {{
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
}}

/* ── Global reset ── */
*, *::before, *::after {{ box-sizing: border-box; }}

body, html {{
  background: var(--h-bg);
  color: var(--h-text);
  font-family: var(--h-font);
  margin: 0;
  padding: 0;
  height: 100vh;
  display: flex;
  flex-direction: column;
}}

body::before {{
  content: "";
  position: fixed;
  inset: 0;
  background:
    radial-gradient(ellipse 800px 400px at 50% -100px, rgba(191, 29, 29, 0.08), transparent 70%),
    radial-gradient(ellipse 600px 300px at 50% 100%, rgba(191, 29, 29, 0.03), transparent 60%);
  pointer-events: none;
  z-index: 0;
}}

#app-container {{
  max-width: 720px;
  width: 100%;
  margin: 0 auto;
  padding: 20px 16px;
  position: relative;
  z-index: 1;
  display: flex;
  flex-direction: column;
  gap: 16px;
}}

/* ── HAL Eye ── */
#hal-hero {{
  text-align: center;
  padding: 40px 16px 8px;
  position: relative;
}}

#hal-eye-outer {{
  width: 160px;
  height: 160px;
  margin: 0 auto 18px;
  position: relative;
  display: grid;
  place-items: center;
}}

/* Subtle ring halo */
#hal-eye-outer::before {{
  content: "";
  position: absolute;
  inset: -8px;
  border-radius: 50%;
  border: 1px solid rgba(191, 29, 29, 0.08);
}}

#hal-eye-shell {{
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
}}

#hal-eye {{
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
}}

/* Glass highlight */
#hal-eye::before {{
  content: "";
  position: absolute;
  top: 12%;
  left: 20%;
  width: 30%;
  height: 30%;
  background: radial-gradient(circle, rgba(255,255,255,0.15), transparent 70%);
  border-radius: 50%;
  pointer-events: none;
}}

/* Lens flare streak */
#hal-eye::after {{
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
}}

#hal-eye.off {{
  background: radial-gradient(circle at 36% 30%,
    #606060 0%, #484848 25%, #2a2a2a 55%, #141414 100%);
  box-shadow: 0 0 15px rgba(100, 100, 100, 0.05);
}}
#hal-eye.off::before, #hal-eye.off::after {{ opacity: 0.15; }}

#hal-eye.listening {{
  box-shadow: 0 0 50px rgba(191, 29, 29, 0.5), 0 0 100px rgba(191, 29, 29, 0.15);
  transform: scale(1.015);
  animation: hal-breathe 3s ease-in-out infinite;
}}

#hal-eye.speaking {{
  box-shadow: 0 0 60px rgba(191, 29, 29, 0.7), 0 0 120px rgba(191, 29, 29, 0.25);
  animation: hal-pulse 1.2s ease-in-out infinite;
}}

@keyframes hal-pulse {{
  0%, 100% {{ box-shadow: 0 0 55px rgba(191, 29, 29, 0.6), 0 0 110px rgba(191, 29, 29, 0.2); transform: scale(1.01); }}
  50%      {{ box-shadow: 0 0 80px rgba(191, 29, 29, 0.9), 0 0 150px rgba(191, 29, 29, 0.35); transform: scale(1.025); }}
}}

@keyframes hal-breathe {{
  0%, 100% {{ box-shadow: 0 0 45px rgba(191, 29, 29, 0.4), 0 0 90px rgba(191, 29, 29, 0.1); transform: scale(1.015); }}
  50%      {{ box-shadow: 0 0 60px rgba(191, 29, 29, 0.55), 0 0 120px rgba(191, 29, 29, 0.18); transform: scale(1.03); }}
}}

#hal-title {{
  font-size: 1.1rem;
  letter-spacing: 0.22em;
  color: var(--h-text-bright);
  margin: 0;
  text-transform: uppercase;
  font-weight: 300;
}}

#hal-subtitle {{
  color: var(--h-text-dim);
  font-size: 0.72rem;
  margin-top: 3px;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  font-weight: 400;
}}

/* ── Call Status Bar ── */
#hal-call-bar {{
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 14px;
  padding: 10px 0 14px;
}}

#hal-timer {{
  color: var(--h-text-dim);
  font-size: 0.78rem;
  font-variant-numeric: tabular-nums;
  letter-spacing: 0.05em;
  opacity: 0.7;
}}

#hal-status {{
  font-size: 0.78rem;
  color: var(--h-text-dim);
  letter-spacing: 0.02em;
}}

#hal-status.ok {{ color: var(--h-green); }}
#hal-status.warn {{ color: var(--h-yellow); }}
#hal-status.error {{ color: var(--h-err); }}

/* ── Mic Level ── */
#hal-level-wrap {{
  display: none;
  align-items: center;
  gap: 10px;
  padding: 4px 0 8px;
  margin: 0 auto;
  max-width: 320px;
  width: 100%;
}}

#hal-level-label {{
  font-size: 0.62rem;
  color: var(--h-text-dim);
  letter-spacing: 0.1em;
  text-transform: uppercase;
}}

#hal-level-track {{
  flex: 1;
  height: 3px;
  border-radius: 2px;
  background: var(--h-border);
  overflow: hidden;
}}

#hal-level-fill {{
  height: 100%;
  width: 0%;
  background: linear-gradient(90deg, #6a1a1a, var(--h-red));
  border-radius: 2px;
  transition: width 60ms linear;
}}

/* ── Transcript ── */
#hal-transcript {{
  border: 1px solid var(--h-border-subtle);
  border-radius: var(--h-radius);
  background: var(--h-surface);
  min-height: 220px;
  max-height: 380px;
  overflow-y: auto;
  padding: 16px 18px;
  margin-bottom: 4px;
}}

.hal-transcript-placeholder {{
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 180px;
  gap: 8px;
}}

.hal-tp-icon {{ font-size: 1.8rem; color: #2a1818; line-height: 1; }}
.hal-tp-text {{ color: #3a2828; font-size: 0.76rem; letter-spacing: 0.04em; }}

.hal-transcript-entry {{ margin-bottom: 12px; line-height: 1.45; }}
.hal-transcript-entry:last-child {{ margin-bottom: 0; }}

.hal-role {{
  font-size: 0.65rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  font-weight: 600;
  opacity: 0.85;
}}

.hal-role-agent {{ color: #c45050; }}
.hal-role-caller {{ color: #8a7070; }}

.hal-text {{ margin-top: 3px; font-size: 0.8rem; color: var(--h-text); line-height: 1.5; }}

#hal-transcript::-webkit-scrollbar {{ width: 4px; }}
#hal-transcript::-webkit-scrollbar-track {{ background: transparent; }}
#hal-transcript::-webkit-scrollbar-thumb {{ background: #221818; border-radius: 2px; }}
#hal-transcript::-webkit-scrollbar-thumb:hover {{ background: #3a2828; }}

/* ── Call Buttons ── */
.button-row {{
  display: flex;
  gap: 12px;
  width: 100%;
}}

button {{
  cursor: pointer;
  outline: none;
}}

button.primary {{
  background: linear-gradient(180deg, #5a1616 0%, #3d0e0e 100%);
  border: 1px solid rgba(191, 29, 29, 0.4);
  color: #f0d0d0;
  font-family: var(--h-font);
  border-radius: 10px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  font-size: 0.78rem;
  padding: 14px 24px;
  flex: 2;
  transition: all 200ms ease;
}}

button.primary:hover {{
  border-color: rgba(191, 29, 29, 0.6);
  box-shadow: 0 0 20px rgba(191, 29, 29, 0.15), inset 0 1px 0 rgba(255,255,255,0.03);
  background: linear-gradient(180deg, #6a1c1c 0%, #451111 100%);
}}

button.primary:active {{ transform: scale(0.98); }}

button.stop {{
  background: transparent;
  border: 1px solid #3a2222;
  color: #a07070;
  font-family: var(--h-font);
  border-radius: 10px;
  font-weight: 500;
  font-size: 0.76rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  padding: 14px 24px;
  flex: 1;
  transition: all 200ms ease;
}}

button.stop:hover {{
  border-color: #5a2828;
  color: #d08080;
  background: rgba(191, 29, 29, 0.06);
}}

/* ── Accordions (Readiness Checks) ── */
details {{
  background: var(--h-surface);
  border: 1px solid var(--h-border-subtle);
  border-radius: var(--h-radius);
  overflow: hidden;
  margin-top: 8px;
}}

details summary {{
  color: var(--h-text-dim);
  font-size: 0.72rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  padding: 12px 16px;
  cursor: pointer;
  background: var(--h-surface);
  transition: color 150ms ease;
  user-select: none;
}}

details summary:hover {{ color: var(--h-text); }}
details[open] summary {{ border-bottom: 1px solid var(--h-border-subtle); }}

/* ── Checks formatting ── */
.hal-checks-container {{ padding: 12px 16px; }}
.hal-no-checks {{ color: #3a2828; font-size: 0.76rem; }}

.hal-check-row {{
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 7px 0;
  border-bottom: 1px solid var(--h-border-subtle);
  font-size: 0.74rem;
  line-height: 1.4;
  color: #a08888;
}}
.hal-check-row:last-child {{ border-bottom: none; }}

.hal-check-dot {{
  width: 7px;
  height: 7px;
  border-radius: 50%;
  margin-top: 5px;
  flex-shrink: 0;
}}
.hal-check-dot.ok {{ background: var(--h-green); box-shadow: 0 0 6px rgba(61,184,106,0.15); }}
.hal-check-dot.warn {{ background: var(--h-yellow); box-shadow: 0 0 6px rgba(212,163,44,0.15); }}
.hal-check-dot.error {{ background: var(--h-err); box-shadow: 0 0 6px rgba(224,85,85,0.15); }}

.hal-check-title {{
  font-size: 0.68rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #8a6a6a;
  margin-right: 6px;
  font-weight: 500;
}}
.hal-check-msg {{ color: #6a5555; }}
  </style>
</head>
<body>

<div id="app-container">
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
    <span id="hal-status" class="{tone}">{summary_text}</span>
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

  <div class="button-row">
    <button class="primary" id="btn-start" onclick="window.HalAudio && window.HalAudio.startCall()">Start Demo Call</button>
    <button class="stop" id="btn-stop" onclick="window.HalAudio && window.HalAudio.endCall()">End Call</button>
  </div>

  <details>
    <summary>Readiness Checks</summary>
    {checks_html}
  </details>
  <details>
    <summary>Settings</summary>
    <div style="padding: 12px 16px; font-size: 0.74rem; color: var(--h-text-dim);">
      To modify the server configuration, edit the <code>.env</code> file in the application's root directory and restart the server.
    </div>
  </details>
</div>

<script>
(() => {{
  // ── Mu-law codec (ITU-T G.711) ──
  const MULAW_BIAS = 0x84;
  const MULAW_CLIP = 32635;
  const MULAW_EXP_TABLE = [0, 132, 396, 924, 1980, 4092, 8316, 16764];

  function linearToMulaw(sample) {{
    sample = Math.max(-32768, Math.min(32767, Math.round(sample)));
    const sign = sample < 0 ? 0x80 : 0;
    if (sample < 0) sample = -sample;
    sample = Math.min(sample, MULAW_CLIP);
    sample += MULAW_BIAS;
    let exponent = 7;
    for (let expMask = 0x4000; exponent > 0; exponent--, expMask >>= 1) {{
      if (sample >= expMask) break;
    }}
    const mantissa = (sample >> (exponent + 3)) & 0x0f;
    return ~(sign | (exponent << 4) | mantissa) & 0xff;
  }}

  function mulawToLinear(mulaw) {{
    mulaw = ~mulaw & 0xff;
    const sign = mulaw & 0x80;
    const exponent = (mulaw >> 4) & 0x07;
    const mantissa = mulaw & 0x0f;
    let sample = MULAW_EXP_TABLE[exponent] + (mantissa << (exponent + 3));
    if (sign) sample = -sample;
    return sample;
  }}

  // ── Resampling ──
  function downsample(float32Buf, fromRate, toRate) {{
    const ratio = fromRate / toRate;
    const outLen = Math.floor(float32Buf.length / ratio);
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {{
      const srcIdx = i * ratio;
      const floor_ = Math.floor(srcIdx);
      const ceil_ = Math.min(floor_ + 1, float32Buf.length - 1);
      const frac = srcIdx - floor_;
      out[i] = float32Buf[floor_] * (1 - frac) + float32Buf[ceil_] * frac;
    }}
    return out;
  }}

  function upsample(int16Samples, fromRate, toRate) {{
    const ratio = toRate / fromRate;
    const outLen = Math.round(int16Samples.length * ratio);
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {{
      const srcIdx = i / ratio;
      const floor_ = Math.floor(srcIdx);
      const ceil_ = Math.min(floor_ + 1, int16Samples.length - 1);
      const frac = srcIdx - floor_;
      const sample = int16Samples[floor_] * (1 - frac) + int16Samples[ceil_] * frac;
      out[i] = sample / 32768;
    }}
    return out;
  }}

  // ── HalAudio namespace ──
  const TARGET_RATE = 8000;
  const FRAME_SIZE = 160;

  window.HalAudio = {{
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

    setEye(state) {{
      const eye = document.getElementById("hal-eye");
      if (eye) eye.className = state;
    }},

    setStatus(text, tone) {{
      const el = document.getElementById("hal-status");
      if (el) {{
        el.textContent = text;
        el.className = tone || "warn";
      }}
    }},

    setTimer(text) {{
      const el = document.getElementById("hal-timer");
      if (el) el.textContent = text;
    }},

    setMicLevel(pct) {{
      const bar = document.getElementById("hal-level-fill");
      if (bar) bar.style.width = pct + "%";
      const wrap = document.getElementById("hal-level-wrap");
      if (wrap) wrap.style.display = pct >= 0 ? "flex" : "none";
    }},

    appendTranscript(role, text) {{
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
    }},

    startTimer() {{
      this.stopTimer();
      this.callStartedAtMs = Date.now();
      this.setTimer("00:00");
      this.callTimerHandle = setInterval(() => {{
        const ms = Date.now() - this.callStartedAtMs;
        const totalSec = Math.max(0, Math.floor(ms / 1000));
        const min = String(Math.floor(totalSec / 60)).padStart(2, "0");
        const sec = String(totalSec % 60).padStart(2, "0");
        this.setTimer(min + ":" + sec);
      }}, 250);
    }},

    stopTimer() {{
      if (this.callTimerHandle) {{
        clearInterval(this.callTimerHandle);
        this.callTimerHandle = null;
      }}
      this.setTimer("00:00");
    }},

    queuePlayback(mulawBase64) {{
      if (!this.audioCtx) return;
      const binary = atob(mulawBase64);
      const mulawBytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) {{
        mulawBytes[i] = binary.charCodeAt(i);
      }}
      const int16Samples = new Int16Array(mulawBytes.length);
      for (let i = 0; i < mulawBytes.length; i++) {{
        int16Samples[i] = mulawToLinear(mulawBytes[i]);
      }}
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
      const entry = {{ source }};
      this.activeSources.push(entry);
      source.onended = () => {{
        const idx = this.activeSources.indexOf(entry);
        if (idx !== -1) this.activeSources.splice(idx, 1);
        if (this.activeSources.length === 0) {{
          this.isPlaying = false;
          if (this.isCallActive) {{
            this.setEye("listening");
            this.setStatus("Listening...", "ok");
          }}
        }}
      }};
    }},

    clearPlayback() {{
      for (const s of this.activeSources) {{
        try {{ s.source.stop(); }} catch (_) {{}}
      }}
      this.activeSources = [];
      this.playbackTime = 0;
      this.isPlaying = false;
    }},

    onAudioProcess(e) {{
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

      while (this.micBuffer.length >= FRAME_SIZE) {{
        const frame = this.micBuffer.slice(0, FRAME_SIZE);
        this.micBuffer = this.micBuffer.slice(FRAME_SIZE);
        const mulawBytes = new Uint8Array(FRAME_SIZE);
        for (let i = 0; i < FRAME_SIZE; i++) {{
          mulawBytes[i] = linearToMulaw(frame[i] * 32768);
        }}
        let binary = "";
        for (let i = 0; i < mulawBytes.length; i++) {{
          binary += String.fromCharCode(mulawBytes[i]);
        }}
        this.ws.send(JSON.stringify({{ event: "media", media: {{ payload: btoa(binary) }} }}));
      }}
    }},

    async startCall() {{
      if (this.isCallActive) return;

      // Clear transcript
      const box = document.getElementById("hal-transcript");
      if (box) box.innerHTML = "";

      this.setStatus("Requesting microphone access...", "warn");

      // Browsers kill navigator.mediaDevices on non-localhost HTTP.
      // Auto-redirect to localhost if on LAN IP (but NOT if already on HTTPS).
      if (!navigator.mediaDevices && location.protocol !== "https:") {{
        const port = location.port || "8080";
        const localhostUrl = "http://localhost:" + port + location.pathname;
        this.setStatus("Redirecting to localhost for mic access...", "warn");
        window.location.href = localhostUrl;
        return;
      }}

      try {{
        this.micStream = await navigator.mediaDevices.getUserMedia({{
          audio: {{ echoCancellation: true, noiseSuppression: true, autoGainControl: true }}
        }});
      }} catch (err) {{
        const msg = (err.name === "NotAllowedError")
          ? "Microphone denied. Allow mic and try again."
          : (err.name === "NotFoundError")
            ? "No microphone found."
            : "Mic error: " + err.message;
        this.setStatus(msg, "error");
        return;
      }}

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

      this.ws.onopen = () => {{
        this.startTimer();
        this.setStatus("Connected. HAL is greeting you...", "ok");
        this.setEye("speaking");
      }};

      this.ws.onmessage = (event) => {{
        const data = JSON.parse(event.data);
        if (data.event === "media") {{
          this.setEye("speaking");
          this.setStatus("HAL is speaking...", "ok");
          this.queuePlayback(data.media.payload);
        }} else if (data.event === "clear") {{
          this.clearPlayback();
        }} else if (data.event === "transcript") {{
          this.appendTranscript(data.role, data.text);
        }} else if (data.event === "hangup") {{
          this.setStatus("HAL ended the call.", "warn");
          this.setEye("off");
          this.cleanup();
          return;
        }}
      }};

      this.ws.onclose = (event) => {{
        if (this.isCallActive) {{
          const reason = event.reason ? " (" + event.reason + ")" : "";
          this.setStatus("Call ended" + reason + ".", "warn");
          this.setEye("off");
          this.cleanup();
        }}
      }};

      this.ws.onerror = () => {{
        this.setStatus("WebSocket error. Is server running?", "error");
        this.setEye("off");
        this.cleanup();
      }};
    }},

    endCall() {{
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {{
        this.ws.send(JSON.stringify({{ event: "stop" }}));
      }}
      this.setStatus("Ending call...", "warn");
      this.setEye("off");
      this.cleanup();
    }},

    cleanup() {{
      this.isCallActive = false;
      this.stopTimer();
      if (this.scriptNode) {{ this.scriptNode.disconnect(); this.scriptNode = null; }}
      if (this.micSource) {{ this.micSource.disconnect(); this.micSource = null; }}
      if (this.micStream) {{ this.micStream.getTracks().forEach(t => t.stop()); this.micStream = null; }}
      this.clearPlayback();
      if (this.ws) {{ try {{ this.ws.close(); }} catch (_) {{}} this.ws = null; }}
      if (this.audioCtx) {{ try {{ this.audioCtx.close(); }} catch (_) {{}} this.audioCtx = null; }}
      this.setMicLevel(-1);
    }}
  }};

  // Initialize eye state
  window.HalAudio.setEye("off");
  console.log("[HAL] Audio pipeline initialized natively");
}})();
</script>
</body>
</html>"""
