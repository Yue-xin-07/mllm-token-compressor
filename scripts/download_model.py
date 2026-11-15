#!/usr/bin/env python3
"""Download a Hugging Face model snapshot into the AutoDL data disk."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_LOCAL_DIR = Path("/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a model snapshot with Hugging Face Hub authentication. "
            "Interrupted downloads can be resumed by running the command again."
        )
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face repository ID.")
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=DEFAULT_LOCAL_DIR,
        help=f"Destination directory (default: {DEFAULT_LOCAL_DIR}).",
    )
    parser.add_argument("--revision", default=None, help="Optional branch, tag, or commit hash.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "huggingface_hub is not installed. Run: "
            "uv pip install --python .venv/bin/python huggingface_hub",
            file=sys.stderr,
        )
        return 2

    args.local_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.model_id} to {args.local_dir}")
    print("Hugging Face credentials are read from the local CLI login or HF_TOKEN.")
    try:
        downloaded_path = snapshot_download(
            repo_id=args.model_id,
            local_dir=str(args.local_dir),
            revision=args.revision,
        )
    except Exception as exc:
        print(f"Model download failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"Model snapshot ready at: {downloaded_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
