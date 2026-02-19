#!/usr/bin/env python3
"""Download Chatterbox Turbo weights into a local bundle directory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download
import chatterbox.tts_turbo as chatterbox_turbo

ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"]
REQUIRED_FILES = ("ve.safetensors", "t3_turbo_v1.safetensors", "s3gen_meanflow.safetensors")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prefetch Chatterbox model bundle for offline use.")
    parser.add_argument(
        "--output",
        default="models/chatterbox",
        help="Output directory for bundled model files (default: models/chatterbox)",
    )
    parser.add_argument(
        "--token",
        default="",
        help="Optional HF token; falls back to HF_TOKEN env var",
    )
    args = parser.parse_args()

    out_dir = Path(args.output).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    token = (args.token or os.getenv("HF_TOKEN", "")).strip() or False
    print(f"Downloading {chatterbox_turbo.REPO_ID} into {out_dir}")
    if token is False:
        print("HF token: not set (anonymous/public download)")
    else:
        print("HF token: provided")

    snapshot_download(
        repo_id=chatterbox_turbo.REPO_ID,
        token=token,
        local_dir=str(out_dir),
        allow_patterns=ALLOW_PATTERNS,
    )

    missing = [name for name in REQUIRED_FILES if not (out_dir / name).is_file()]
    if missing:
        print(f"ERROR: bundle incomplete, missing files: {', '.join(missing)}")
        return 1

    print("Bundle ready.")
    print(f"Set TTS_MODEL_DIR={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
