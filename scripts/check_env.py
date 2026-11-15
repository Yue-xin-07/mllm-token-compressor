#!/usr/bin/env python3
"""Inspect the ML environment and optionally run a Qwen2.5-VL smoke test."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import platform
import sys
from pathlib import Path

DEFAULT_MODEL_PATH = Path("/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct")
PACKAGES = (
    ("PyTorch", "torch", "torch"),
    ("Transformers", "transformers", "transformers"),
    ("FlashAttention-2", "flash-attn", "flash_attn"),
    ("vLLM", "vllm", "vllm"),
    ("Accelerate", "accelerate", "accelerate"),
)


def package_status(label: str, distribution: str, module: str) -> tuple[bool, str]:
    try:
        importlib.util.find_spec(module)
        version = importlib.metadata.version(distribution)
        print(f"{label}: installed ({version})")
        return True, version
    except (importlib.metadata.PackageNotFoundError, ModuleNotFoundError, ValueError):
        print(f"{label}: not installed")
        return False, ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report the Python/CUDA inference environment. Without --no-model-test, "
            "load the configured Qwen2.5-VL model and generate a short text response."
        )
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Local model directory (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--no-model-test",
        action="store_true",
        help="Only report environment status; do not load model weights.",
    )
    return parser.parse_args()


def smoke_test(model_path: Path) -> None:
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run the full check on a GPU instance.")
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")

    print(f"Loading model from {model_path} (bfloat16, automatic device map)...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(str(model_path))
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Reply with the single word OK."}],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=16, do_sample=False)

    prompt_length = inputs["input_ids"].shape[1]
    response = processor.batch_decode(
        output_ids[:, prompt_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    if not response:
        raise RuntimeError("Model loaded but produced an empty response.")
    print(f"Model forward/generation succeeded: {response!r}")


def main() -> int:
    args = parse_args()
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform: {platform.platform()}")

    installed: dict[str, bool] = {}
    for label, distribution, module in PACKAGES:
        present, _ = package_status(label, distribution, module)
        installed[module] = present

    cuda_available = False
    if installed.get("torch"):
        try:
            import torch

            cuda_available = torch.cuda.is_available()
            print(f"PyTorch CUDA build: {torch.version.cuda or 'CPU-only'}")
            print(f"CUDA available: {cuda_available}")
            if cuda_available:
                for index in range(torch.cuda.device_count()):
                    print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
        except Exception as exc:
            print(f"PyTorch CUDA check failed: {type(exc).__name__}: {exc}")
    else:
        print("CUDA available: unknown (PyTorch is not installed)")

    if args.no_model_test:
        print("Model smoke test: skipped (--no-model-test)")
        return 0

    missing = [name for name in ("torch", "transformers", "flash_attn", "vllm") if not installed[name]]
    if missing:
        print(f"Required GPU packages are missing: {', '.join(missing)}", file=sys.stderr)
        return 1
    if not cuda_available:
        print("CUDA is unavailable; use a GPU-enabled AutoDL instance.", file=sys.stderr)
        return 1

    try:
        smoke_test(args.model_path)
    except Exception as exc:
        print(f"Model smoke test failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
