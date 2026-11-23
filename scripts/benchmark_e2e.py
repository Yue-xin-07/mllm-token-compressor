"""Run one side of the Step 4 benchmark in a fresh process."""
import argparse
from functools import partial
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_workloads import measure, workloads
from compressor.vllm_engine import CompressedEngine


MODEL = "/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct"
MAX_MODEL_LEN = 32800
MAX_IMAGE_PIXELS = 448 * 448


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("native", "compressed"))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.95)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be nonnegative and repeats must be positive")

    from vllm import SamplingParams, __version__ as vllm_version
    import torch

    config = {
        "model": args.model,
        "max_model_len": MAX_MODEL_LEN,
        "max_num_seqs": 2,
        "max_image_pixels": MAX_IMAGE_PIXELS,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "gpu": torch.cuda.get_device_name(),
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "torch": str(torch.__version__),
        "vllm": vllm_version,
        "dtype": "bfloat16",
        "max_output_tokens": 32,
    }
    data = {"mode": args.mode, "config": config, "cases": {}}
    sampling = SamplingParams(temperature=0, max_tokens=32, ignore_eos=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    engine = CompressedEngine(args.model, MAX_MODEL_LEN,
                              args.gpu_memory_utilization, max_num_seqs=2,
                              max_image_pixels=MAX_IMAGE_PIXELS)
    run = engine.generate_native_batch if args.mode == "native" else engine.generate_batch
    try:
        for name, prompts, expected in workloads(engine):
            rounds = []
            errors = []
            for trial in range(-args.warmup, args.repeats):
                try:
                    stats = measure(partial(run, prompts, sampling), trial >= 0)
                    observed = [row["original_tokens"] for row in stats["requests"]]
                    if observed != expected or stats["max_scheduled_requests"] != len(expected):
                        raise RuntimeError(f"unexpected input lengths or scheduler batch: {stats}")
                    if [row["output_tokens"] for row in stats["requests"]] != [32] * len(expected):
                        raise RuntimeError("request did not produce 32 output tokens")
                except RuntimeError as error:
                    errors.append({"round": trial, "error": str(error)})
                    if engine.engine.has_unfinished_requests():
                        raise
                else:
                    if trial >= 0:
                        rounds.append(stats)
            data["cases"][name] = {"input_lengths": expected, "rounds": rounds, "errors": errors}
            args.output.write_text(json.dumps(data, indent=2) + "\n")
            print(f"{args.mode} {name}: {len(rounds)} measured, {len(errors)} errors", flush=True)
    finally:
        engine.close()
    if any(case["errors"] for case in data["cases"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
