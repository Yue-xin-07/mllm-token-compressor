"""Create the final Step 4 Markdown report from matching run artifacts."""
import argparse
from collections import defaultdict
import json
from math import ceil
from pathlib import Path
from statistics import median


GIB = 1024 ** 3


def read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def paired(root, prefix):
    native = read(root / f"{prefix}native.json")
    compressed = read(root / f"{prefix}compressed.json")
    if native and compressed:
        if native["config"] != compressed["config"]:
            raise ValueError(f"{prefix}configs differ")
        if (native["mode"], compressed["mode"]) != ("native", "compressed"):
            raise ValueError(f"{prefix}unexpected backend in results")
    return native, compressed


def perf_stats(rounds):
    ttft = sorted(row["ttft_ms"] for trial in rounds for row in trial["requests"])
    tokens = sum(row["output_tokens"] for trial in rounds for row in trial["requests"])
    process_memory = max((trial.get("peak_process_memory_bytes") or 0 for trial in rounds))
    return [
        median(ttft),
        ttft[ceil(.95 * len(ttft)) - 1],
        tokens / sum(trial["elapsed_seconds"] for trial in rounds),
        max(trial["peak_device_memory_bytes"] for trial in rounds) / GIB,
        process_memory / GIB if process_memory else None,
        max(trial["peak_torch_reserved_bytes"] for trial in rounds) / GIB,
        max(trial["peak_torch_allocated_bytes"] for trial in rounds) / GIB,
        max(trial["kv_pool_bytes"] for trial in rounds) / GIB,
        max(trial["kv_peak_bytes"] for trial in rounds) / GIB,
    ]


def performance(native, compressed):
    lines = [
        "## Performance",
        "",
        "TTFT uses burst submission to first returned token. Throughput divides total output "
        "tokens by total measured wall time. GPU device and process peaks are sampled by NVML every "
        "10 ms. PyTorch peaks are process-local allocator metrics. KV occupancy comes from "
        "the scheduler after block allocation and before generated results can free blocks.",
        "",
    ]
    if not (native and compressed):
        return lines + ["A matched performance run is unavailable.", ""], {}
    config = native["config"]
    lines += [
        f"{config['gpu']}, PyTorch {config['torch']}, vLLM {config['vllm']} V0, "
        f"{config['dtype']}, GPU memory utilization {config['gpu_memory_utilization']}. "
        f"Each backend ran in a fresh process with {config['warmup']} warm-up rounds "
        f"and {config['repeats']} measured rounds per workload. Image cap "
        f"{config['max_image_pixels']:,} pixels. Each response was fixed to 32 tokens.",
        f"The two backends share a {config.get('kv_pool_blocks', 'profile-sized')}-block KV pool "
        f"and allocator configuration `{config.get('cuda_allocator', 'default')}`.",
        "The synthetic long inputs use repeated text and one 448×448 image. The mixed workload "
        "submits a 2K text request and an 8K image request in the same scheduling round.",
        "",
    ]
    labels = ("TTFT p50 (ms)", "TTFT p95 (ms)", "Output tokens/s",
              "Peak device memory (GiB)", "Peak process memory (GiB)", "Peak PyTorch reserved (GiB)",
              "Peak PyTorch allocated (GiB)", "KV pool (GiB)", "Peak occupied KV (GiB)")
    gates = {}
    for name, raw in native["cases"].items():
        other = compressed["cases"].get(name)
        if not other or other["input_lengths"] != raw["input_lengths"]:
            raise ValueError(f"mismatched workload {name}")
        base, pruned = raw["rounds"], other["rounds"]
        lines += [f"### {name}", "",
                  f"Measured rounds, native {len(base)}, compressed {len(pruned)}. "
                  f"Failed rounds, native {len(raw['errors'])}, compressed {len(other['errors'])}.", ""]
        if not base or not pruned:
            lines += ["A matched result is unavailable for this workload.", ""]
            continue
        a, b = perf_stats(base), perf_stats(pruned)
        lines += ["| Metric | Native | Compressed | Change |", "| --- | ---: | ---: | ---: |"]
        for label, x, y in zip(labels, a, b):
            left = "Unavailable" if x is None else f"{x:.3f}"
            right = "Unavailable" if y is None else f"{y:.3f}"
            percent = "—" if not x or y is None else f"{(y / x - 1) * 100:+.2f}%"
            lines.append(f"| {label} | {left} | {right} | {percent} |")
        rows = pruned[0]["requests"]
        vision = [row["visual_tokens"] for row in rows]
        removable = sum(value - ceil(.25 * value) for value in vision)
        original = sum(raw["input_lengths"])
        retained = [row["retained_tokens"] for row in rows]
        native_blocks = max(trial["kv_peak_blocks"] for trial in base)
        compressed_blocks = max(trial["kv_peak_blocks"] for trial in pruned)
        lines += [
            "",
            f"Input tokens by request {raw['input_lengths']}, retained {retained}, "
            f"visual {vision}. Peak KV blocks {native_blocks:,} → {compressed_blocks:,}. "
            f"The 25% minimum visual retention caps input token removal at "
            f"{removable:,} of {original:,} ({100 * removable / original:.2f}%) "
            "before generated tokens and cache block rounding.",
            "",
        ]
        if (name == "single_32k" and config["warmup"] >= 3 and config["repeats"] >= 20
                and len(base) == len(pruned) == config["repeats"]
                and not raw["errors"] and not other["errors"]):
            gates = {"TTFT": 100 * (1 - b[0] / a[0]), "KV": 100 * (1 - b[-1] / a[-1])}
    return lines, gates


def accuracy(native, compressed):
    lines = ["## ScreenSpot click accuracy", "",
             "Original [SeeClick test set](https://github.com/njucckevin/SeeClick#gui-grounding-benchmark-screenspot). "
             "A click on or inside the target box is correct. Missing or invalid points count as errors. "
             "Both backends use the same normalized-coordinate prompt and deterministic decoding.", ""]
    if not (native and compressed):
        return lines + ["A matched accuracy run is unavailable.", ""], None
    config = native["config"]
    a, b = native["rows"], compressed["rows"]
    count = config["full_count"]
    common = sorted(a.keys() & b.keys())
    lines += [f"Native {len(a):,}/{count:,}, compressed {len(b):,}/{count:,}, paired {len(common):,}.",
              ""]
    lines.extend(f"- `{name}` SHA-256 `{digest}`"
                 for name, digest in config["annotations_sha256"].items())
    if config.get("archive_sha256"):
        lines.append(f"- `screenspot_imgs.tar` SHA-256 `{config['archive_sha256']}`")
    lines.append("")
    if not common:
        return lines, None
    groups = defaultdict(list)
    for key in common:
        row = a[key]
        groups[f"{row['split']} / {row['type']}"].append(key)
    groups["Overall"] = common
    overall_delta = None
    lines += ["| Group | Count | Native | Compressed | Delta (pp) |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for name, keys in groups.items():
        x = 100 * sum(a[key]["correct"] for key in keys) / len(keys)
        y = 100 * sum(b[key]["correct"] for key in keys) / len(keys)
        lines.append(f"| {name} | {len(keys):,} | {x:.2f}% | {y:.2f}% | {y - x:+.2f} |")
        if name == "Overall" and len(keys) == count:
            overall_delta = y - x
    invalid = [sum(rows[key]["point"] is None for key in common) for rows in (a, b)]
    lines += ["", f"Invalid outputs on paired samples, native {invalid[0]}, compressed {invalid[1]}.", ""]
    return lines, overall_delta


def grade(observed, threshold, unit):
    if observed is None:
        return "Unverified"
    status = "Met" if observed >= threshold else "Missed"
    return f"{status} ({observed:+.2f}{unit})"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("outputs/step4"))
    parser.add_argument("--output", type=Path, default=Path("evaluation_report.md"))
    args = parser.parse_args()
    perf, gates = performance(*paired(args.results, ""))
    acc, delta = accuracy(*paired(args.results, "screenspot-"))
    lines = [
        "# Step 4 evaluation report",
        "",
        "| Project target | Outcome |",
        "| --- | --- |",
        f"| 32K TTFT p50 reduction ≥35% | {grade(gates.get('TTFT'), 35, '%')} |",
        f"| 32K peak occupied KV reduction ≥40% | {grade(gates.get('KV'), 40, '%')} |",
        f"| ScreenSpot accuracy delta ≥−1 pp | {grade(delta, -1, ' pp')} |",
        "",
        *perf, *acc,
        "## Reproduction",
        "",
        "Run from the repository root with the existing AutoDL environment. Download the original "
        "[ScreenSpot release](https://box.nju.edu.cn/d/5b8892c1901c4dbeb715/) into `data/screenspot`.",
        "",
        "```bash",
        "mkdir -p data/screenspot",
        "base=https://box.nju.edu.cn/d/5b8892c1901c4dbeb715/files/",
        "for file in screenspot_desktop.json screenspot_mobile.json screenspot_web.json screenspot_imgs.tar; do",
        '  curl -fL --retry 2 "$base?p=%2F$file&dl=1" -o "data/screenspot/$file"',
        "done",
        "tar -xf data/screenspot/screenspot_imgs.tar -C data/screenspot",
        "",
        "for mode in native compressed; do",
        "  .venv/bin/python scripts/benchmark_e2e.py --mode \"$mode\" --output \"outputs/step4/$mode.json\"",
        "  .venv/bin/python scripts/evaluate_screenspot.py --mode \"$mode\" --dataset data/screenspot \\",
        "    --output \"outputs/step4/screenspot-$mode.json\"",
        "done",
        ".venv/bin/python scripts/report_evaluation.py",
        "```",
        "",
        f"Raw round and per-sample results are in `{args.results}` on AutoDL.",
        "",
    ]
    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
