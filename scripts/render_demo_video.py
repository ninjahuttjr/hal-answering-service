#!/usr/bin/env python3
"""Render a demo video from a call recording + timeline JSON.

Shows the HAL 9000 eye with a subtle red glow driven by the audio waveform
when HAL is speaking. The glow intensity follows the actual speech amplitude
for a natural, film-accurate look.

The timeline JSON is produced by demo_call.py alongside the WAV.

Requirements:
    - Pillow (pip install Pillow)
    - numpy, soundfile (already in project deps)
    - ffmpeg on PATH

Usage:
    python scripts/render_demo_video.py \
        --recording recordings/demo_call.wav \
        --timeline recordings/demo_call_timeline.json \
        --eye bqqaj4r.png \
        --output demo_scammer.mp4
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageFilter


# ── Video parameters ──
FPS = 30
WIDTH = 1280
HEIGHT = 720
BG_COLOR = (8, 6, 6)

# ── Glow parameters ──
GLOW_COLOR = (191, 29, 29)
GLOW_RADIUS_FACTOR = 3.5       # large region so the gaussian tail fully fades to zero
GLOW_BLUR = 60                 # heavy blur to eliminate any hard edges
GLOW_MAX_INTENSITY = 0.8       # peak glow at loudest speech
GLOW_MIN_INTENSITY = 0.0       # no glow when silent

# RMS smoothing for waveform-driven glow
RMS_WINDOW_MS = 80             # window size for RMS calculation
RMS_SMOOTH_ATTACK = 0.15       # how fast glow ramps up (0-1, higher = faster)
RMS_SMOOTH_RELEASE = 0.06      # how fast glow fades out (0-1, higher = faster)

# Discrete glow levels to pre-bake
GLOW_LEVELS = 48


def load_timeline(timeline_path: Path) -> list[tuple[float, float, str]]:
    with open(timeline_path, encoding="utf-8") as f:
        data = json.load(f)
    sr = data["sample_rate"]
    return [
        (seg["start_sample"] / sr, seg["end_sample"] / sr, seg["role"])
        for seg in data["segments"]
    ]


def get_role_at_time(t: float, segments: list[tuple[float, float, str]]) -> str | None:
    for start, end, role in segments:
        if start <= t <= end:
            return role
    return None


def compute_rms_envelope(audio: np.ndarray, sr: int, fps: int,
                         total_frames: int) -> np.ndarray:
    """Compute per-frame RMS envelope from audio, normalized to 0..1."""
    window_samples = int(sr * RMS_WINDOW_MS / 1000)
    half_win = window_samples // 2
    envelope = np.zeros(total_frames, dtype=np.float32)

    for fi in range(total_frames):
        center = int(fi / fps * sr)
        start = max(0, center - half_win)
        end = min(len(audio), center + half_win)
        if end > start:
            chunk = audio[start:end]
            envelope[fi] = np.sqrt(np.mean(chunk ** 2))

    # Normalize to 0..1 based on the audio's own dynamic range
    peak = np.percentile(envelope[envelope > 0], 95) if np.any(envelope > 0) else 1.0
    if peak > 0:
        envelope = np.clip(envelope / peak, 0.0, 1.0)
    return envelope


def build_frame_cache(eye_path: Path, eye_size: int) -> list[bytes]:
    """Pre-render GLOW_LEVELS frames as raw RGB bytes.

    Level 0 = no glow (silent), level GLOW_LEVELS-1 = max glow (loudest speech).
    """
    # Load + resize + square-crop eye
    eye_raw = Image.open(eye_path).convert("RGB")
    ratio = max(eye_size / eye_raw.width, eye_size / eye_raw.height)
    eye_raw = eye_raw.resize((int(eye_raw.width * ratio), int(eye_raw.height * ratio)), Image.LANCZOS)
    left = (eye_raw.width - eye_size) // 2
    top = (eye_raw.height - eye_size) // 2
    eye_raw = eye_raw.crop((left, top, left + eye_size, top + eye_size))

    # Generate alpha from luminance — dark pixels (black background) become transparent
    # so the glow shows through behind the eye housing
    eye_rgb_arr = np.array(eye_raw, dtype=np.float32) / 255.0  # H x W x 3
    luminance = 0.299 * eye_rgb_arr[..., 0] + 0.587 * eye_rgb_arr[..., 1] + 0.114 * eye_rgb_arr[..., 2]
    # Smooth ramp: fully transparent below threshold, fully opaque above
    alpha_threshold = 0.08  # pixels darker than this are fully transparent
    alpha_ceiling = 0.25    # pixels brighter than this are fully opaque
    alpha = np.clip((luminance - alpha_threshold) / (alpha_ceiling - alpha_threshold), 0.0, 1.0)
    eye_arr = np.dstack([eye_rgb_arr, alpha])  # H x W x 4

    glow_size = int(eye_size * GLOW_RADIUS_FACTOR)

    # Build radial glow mask — gaussian falloff, tight around center
    gy, gx = np.ogrid[:glow_size, :glow_size]
    cx, cy = glow_size // 2, glow_size // 2
    dist = np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2).astype(np.float32)
    sigma = glow_size * 0.18  # controls spread — gaussian dies off well before the edge
    glow_mask = np.exp(-(dist ** 2) / (2 * sigma ** 2))

    # Compute layout positions (centered)
    glow_x = (WIDTH - glow_size) // 2
    glow_y = (HEIGHT - glow_size) // 2
    eye_x = (WIDTH - eye_size) // 2
    eye_y = (HEIGHT - eye_size) // 2

    # Build base background
    bg = np.full((HEIGHT, WIDTH, 3), BG_COLOR, dtype=np.float32) / 255.0

    # Glow region slices on canvas
    gy_start = max(0, glow_y)
    gy_end = min(HEIGHT, glow_y + glow_size)
    gx_start = max(0, glow_x)
    gx_end = min(WIDTH, glow_x + glow_size)
    fg_ys = gy_start - glow_y
    fg_ye = fg_ys + (gy_end - gy_start)
    fg_xs = gx_start - glow_x
    fg_xe = fg_xs + (gx_end - gx_start)

    glow_rgb = np.array(GLOW_COLOR, dtype=np.float32) / 255.0

    # Eye region slices
    ey_start = max(0, eye_y)
    ey_end = min(HEIGHT, eye_y + eye_size)
    ex_start = max(0, eye_x)
    ex_end = min(WIDTH, eye_x + eye_size)
    eye_crop = eye_arr[ey_start - eye_y:ey_end - eye_y, ex_start - eye_x:ex_end - eye_x]
    eye_rgb = eye_crop[..., :3]
    eye_alpha = eye_crop[..., 3:4]

    # Pre-blur the glow mask for soft edges
    glow_mask_pil = Image.fromarray((glow_mask * 255).astype(np.uint8))
    glow_mask_pil = glow_mask_pil.filter(ImageFilter.GaussianBlur(radius=GLOW_BLUR))
    glow_mask_blurred = np.array(glow_mask_pil, dtype=np.float32) / 255.0
    glow_mask_blurred_crop = glow_mask_blurred[fg_ys:fg_ye, fg_xs:fg_xe, np.newaxis]

    cache: list[bytes] = []

    for level_idx in range(GLOW_LEVELS):
        t = level_idx / max(1, GLOW_LEVELS - 1)
        intensity = GLOW_MIN_INTENSITY + (GLOW_MAX_INTENSITY - GLOW_MIN_INTENSITY) * t

        frame = bg.copy()

        # Additive glow — soft red halo behind the eye
        if intensity > 0.001:
            glow_contribution = glow_rgb * (glow_mask_blurred_crop * intensity)
            frame[gy_start:gy_end, gx_start:gx_end] = np.clip(
                frame[gy_start:gy_end, gx_start:gx_end] + glow_contribution, 0.0, 1.0
            )

        # Composite eye on top
        frame[ey_start:ey_end, ex_start:ex_end] = (
            eye_rgb * eye_alpha + frame[ey_start:ey_end, ex_start:ex_end] * (1 - eye_alpha)
        )

        cache.append(np.ascontiguousarray((frame * 255), dtype=np.uint8).tobytes())

    return cache


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render a demo video with HAL eye glow from a call recording."
    )
    parser.add_argument("--recording", required=True, help="Path to call recording WAV")
    parser.add_argument("--timeline", required=True, help="Path to timeline JSON from demo_call.py")
    parser.add_argument("--eye", required=True, help="Path to HAL eye PNG image")
    parser.add_argument("--output", default="demo_video.mp4", help="Output MP4 path")
    parser.add_argument("--fps", type=int, default=FPS, help=f"Frame rate (default: {FPS})")
    parser.add_argument("--eye-size", type=int, default=360, help="Eye diameter in pixels")
    args = parser.parse_args()

    recording_path = Path(args.recording)
    timeline_path = Path(args.timeline)
    eye_path = Path(args.eye)

    for p, label in [(recording_path, "Recording"), (timeline_path, "Timeline"), (eye_path, "Eye image")]:
        if not p.is_file():
            print(f"ERROR: {label} not found: {p}")
            return 1

    if not shutil.which("ffmpeg"):
        print("ERROR: ffmpeg not found on PATH.")
        return 1

    fps = args.fps

    print("Loading audio...", flush=True)
    audio, sr = sf.read(recording_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    duration = len(audio) / sr
    total_frames = int(duration * fps)
    print(f"  {duration:.1f}s -> {total_frames} frames at {fps}fps")

    print("Loading timeline...", flush=True)
    segments = load_timeline(timeline_path)
    print(f"  {len(segments)} segments ({sum(1 for _,_,r in segments if r=='agent')} agent, {sum(1 for _,_,r in segments if r=='caller')} caller)")

    print("Computing audio envelope...", flush=True)
    rms_envelope = compute_rms_envelope(audio, sr, fps, total_frames)

    print(f"Pre-rendering {GLOW_LEVELS} glow levels...", flush=True)
    import time
    t0 = time.perf_counter()
    frame_cache = build_frame_cache(eye_path, args.eye_size)
    cache_mb = sum(len(b) for b in frame_cache) // (1024 * 1024)
    print(f"  Done in {time.perf_counter()-t0:.1f}s. Cache: {cache_mb} MB", flush=True)

    # Build per-frame glow index driven by audio waveform + timeline gating
    frame_indices = np.zeros(total_frames, dtype=np.int32)
    smoothed = 0.0

    for fi in range(total_frames):
        t = fi / fps
        role = get_role_at_time(t, segments)

        # Only glow during HAL (agent) segments, driven by audio amplitude
        if role == "agent":
            target = rms_envelope[fi]
        else:
            target = 0.0

        # Smooth attack/release for natural breathing feel
        if target > smoothed:
            smoothed += (target - smoothed) * RMS_SMOOTH_ATTACK
        else:
            smoothed += (target - smoothed) * RMS_SMOOTH_RELEASE

        frame_indices[fi] = int(smoothed * (GLOW_LEVELS - 1) + 0.5)

    frame_indices = np.clip(frame_indices, 0, GLOW_LEVELS - 1)

    # ── Pipe to ffmpeg ──
    print(f"Encoding {total_frames} frames via ffmpeg...", flush=True)

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{WIDTH}x{HEIGHT}", "-r", str(fps),
        "-i", "pipe:0",
        "-i", str(recording_path),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(args.output),
    ]

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    import threading

    def write_frames():
        try:
            for fi in range(total_frames):
                proc.stdin.write(frame_cache[int(frame_indices[fi])])
                if fi % (fps * 10) == 0 or fi == total_frames - 1:
                    pct = (fi + 1) / total_frames * 100
                    print(f"  {pct:5.1f}% t={fi/fps:.0f}s", flush=True)
        except (BrokenPipeError, OSError):
            pass
        finally:
            proc.stdin.close()

    t0 = time.perf_counter()
    writer = threading.Thread(target=write_frames, daemon=True)
    writer.start()
    stderr_bytes = proc.stderr.read()  # drain stderr so ffmpeg never blocks
    writer.join()
    proc.wait()
    stderr = stderr_bytes.decode(errors="replace")

    if proc.returncode != 0:
        print(f"ERROR: ffmpeg failed (code {proc.returncode}):")
        for line in stderr.strip().splitlines()[-15:]:
            print(f"  {line}")
        return 1

    elapsed = time.perf_counter() - t0
    size_mb = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"Done in {elapsed:.1f}s. Output: {args.output} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
