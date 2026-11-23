"""Measure the Step 2 compression pipeline on Qwen2.5-VL-shaped CUDA inputs."""
import argparse
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compressor.attention_probe import visual_attention
from compressor.scoring import score_attention, score_probe_attention
from compressor.token_pruner import TokenPruner


def measure(operation, warmup, repeats):
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        operation()
        torch.cuda.synchronize()
        samples.append(1000 * (perf_counter() - start))
    return round(statistics.median(samples), 4)


def inputs(queries, visual_tokens):
    length = queries + visual_tokens
    device = "cuda"
    dtype = torch.bfloat16
    q = torch.randn(length, 28, 128, device=device, dtype=dtype)
    k = torch.randn(length, 4, 128, device=device, dtype=dtype)
    indices = torch.arange(length, device=device)
    # Include text on both sides of the visual span to exercise empty causal rows.
    prefix = min(16, queries)
    visual = (indices >= prefix) & (indices < prefix + visual_tokens)
    hidden = torch.randn(length, 3584, device=device, dtype=dtype)
    return q, k, indices[~visual], indices[visual], hidden, indices.expand(3, -1), visual


@torch.inference_mode()
def benchmark(queries, visual_tokens, warmup, repeats):
    q, k, query_indices, visual_indices, hidden, positions, mask = inputs(queries, visual_tokens)
    probe = lambda: visual_attention(q, k, query_indices, visual_indices)
    attention = probe()
    pruner = TokenPruner()
    scoring = lambda: score_attention(attention)
    fused_scoring = lambda: score_probe_attention(attention)
    pruning = lambda: pruner(hidden, positions, mask, attention)
    fused_pruning = lambda: pruner._prune(hidden, positions, mask, attention, score_probe_attention)
    torch.testing.assert_close(fused_pruning().kept_indices, pruning().kept_indices)
    baseline_score = scoring()
    fused_score = fused_scoring()
    torch.testing.assert_close(fused_score.scores, baseline_score.scores, rtol=3e-6, atol=1e-8)
    if fused_score.keep_count != baseline_score.keep_count:
        raise AssertionError("fused scorer changed the visual token budget")
    return {
        "queries": queries,
        "visual_tokens": visual_tokens,
        "probe_ms": measure(probe, warmup, repeats),
        "pytorch_scoring_ms": measure(scoring, warmup, repeats),
        "triton_scoring_ms": measure(fused_scoring, warmup, repeats),
        "pruning_with_pytorch_ms": measure(pruning, warmup, repeats),
        "pruning_with_triton_ms": measure(fused_pruning, warmup, repeats),
    }


def profile_model(model_path):
    from unittest.mock import patch

    from vllm import SamplingParams

    from compressor.vllm_engine import CompressedEngine
    from run_compressed import make_prompt

    torch.cuda.empty_cache()
    adapter = CompressedEngine(model_path, max_model_len=8192)
    prompt = make_prompt(text_repeats=800)
    params = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
    prepare = adapter.prepare
    step = adapter.engine.step
    prepare_ms = []
    step_ms = []

    def timed(operation, samples):
        def run(*args, **kwargs):
            torch.cuda.synchronize()
            start = perf_counter()
            result = operation(*args, **kwargs)
            torch.cuda.synchronize()
            samples.append(1000 * (perf_counter() - start))
            return result
        return run

    full_profile = []
    try:
        adapter.generate(prompt, params)
        with (patch.object(adapter, "prepare", timed(prepare, prepare_ms)),
              patch.object(adapter.engine, "step", timed(step, step_ms))):
            for _ in range(3):
                prepare_ms.clear()
                step_ms.clear()
                adapter.generate(prompt, params)
                full_profile.append((prepare_ms[0], step_ms[0], sum(step_ms[1:])))
        result = {
            "original_tokens": adapter.last_stats["original_tokens"],
            "retained_tokens": adapter.last_stats["retained_tokens"],
            "decode_steps": len(step_ms) - 1,
            "repeats": len(full_profile),
        }
        for label, samples in zip(("prepare_ms", "prefill_step_ms", "decode_total_ms"),
                                  zip(*full_profile)):
            result[label] = round(statistics.median(samples), 4)
        return result
    finally:
        adapter.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", help="also time request preparation, prefill and decode")
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1:
        parser.error("warmup and repeats must be positive")
    torch.manual_seed(42)
    results = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "timing": "Median synchronized CUDA wall time in ms",
        "cases": [benchmark(q, v, args.warmup, args.repeats)
                  for q, v in ((512, 64), (2048, 256), (5623, 64), (2048, 1024))],
    }
    if args.model:
        results["model_profile"] = profile_model(args.model)
    data = json.dumps(results, indent=2)
    if args.output:
        args.output.write_text(data + "\n", encoding="utf-8")
    print(data)


if __name__ == "__main__":
    main()
