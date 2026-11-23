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
from compressor.scoring import score_attention
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


def benchmark(queries, visual_tokens, warmup, repeats):
    q, k, query_indices, visual_indices, hidden, positions, mask = inputs(queries, visual_tokens)
    probe = lambda: visual_attention(q, k, query_indices, visual_indices)
    attention = probe()
    pruner = TokenPruner()
    scoring = lambda: score_attention(attention)
    pruning = lambda: pruner(hidden, positions, mask, attention)
    return {
        "queries": queries,
        "visual_tokens": visual_tokens,
        "probe_ms": measure(probe, warmup, repeats),
        "scoring_ms": measure(scoring, warmup, repeats),
        "pruning_with_scoring_ms": measure(pruning, warmup, repeats),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path)
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
    data = json.dumps(results, indent=2)
    if args.output:
        args.output.write_text(data + "\n", encoding="utf-8")
    print(data)


if __name__ == "__main__":
    main()
