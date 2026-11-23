"""Build the working-set report and the portfolio replay from raw run artifacts."""
import argparse
import json
from pathlib import Path
from random import Random
from statistics import median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.gui_figures import performance_figure, replay_figure
from scripts.report_evaluation import perf_stats
from scripts.screenspot_metrics import load_dataset


def load(root, name):
    return json.loads((root / name).read_text(encoding="utf-8"))


def grade(value, threshold):
    return "Met" if value >= threshold else "Missed"


def summarize_perf(root, destination):
    left, right = (load(root, f"perf-{mode}.json") for mode in ("native", "working_set"))
    if left["config"] != right["config"]:
        raise ValueError("unmatched performance configuration")
    config = left["config"]
    lines = ["## Cold-cache performance", "",
             f"{config['gpu']}, Qwen2.5-VL-7B, BF16, vLLM {config['vllm']} V0. "
             f"Each backend ran in a fresh process with {config['warmup']} warm-ups "
             f"and {config['repeats']} measured rounds per workload. "
             "TTFT includes CPU selection, preprocessing, and the first returned token.", ""]
    labels = ("TTFT p50 (ms)", "TTFT p95 (ms)", "Output tokens/s", "Peak GPU device (GiB)",
              "Peak GPU process (GiB)", "Peak PyTorch reserved (GiB)",
              "Peak PyTorch allocated (GiB)", "KV pool (GiB)", "Peak occupied KV (GiB)")
    gates = {}
    for case in ("single_32k", "mixed_2k_8k"):
        base, proposed = (entry["cases"][case] for entry in (left, right))
        expected = config["repeats"]
        for workload in (base, proposed):
            if len(workload["rounds"]) != expected or workload["errors"]:
                raise ValueError(f"incomplete performance case {case}")
        a, b = (perf_stats(result["rounds"]) for result in (base, proposed))
        rows = [request for trial in proposed["rounds"] for request in trial["requests"]]
        if case == "single_32k":
            gates = {"ttft": 100 * (1 - b[0] / a[0]), "kv": 100 * (1 - b[-1] / a[-1])}
            performance_figure(destination / "gui-performance.png", a, b)
        lines += [f"### {case}", "", "| Metric | Native | Working set |",
                  "| --- | ---: | ---: |"]
        for name, x, y in zip(labels, a, b, strict=True):
            x = f"{x:.3f}" if x is not None else "Unavailable"
            y = f"{y:.3f}" if y is not None else "Unavailable"
            lines.append(f"| {name} | {x} | {y} |")
        lengths = [request["retained_tokens"] for request in proposed["rounds"][0]["requests"]]
        fallback = sum(bool(request["fallback"]) for request in rows)
        lines += ["", f"Retained input tokens by request {lengths}. "
                  f"Planner p50 {median(row['selection_ms'] for row in rows):.2f} ms. "
                  f"Fallbacks {fallback}/{len(rows)}.", ""]
    return lines, gates


def summarize_screenspot(root):
    original = load(root, "screen-native.json")
    selected = load(root, "screen-working_set.json")
    a, b = original["rows"], selected["rows"]
    total = original["config"]["full_count"]
    if (original["config"] != selected["config"] or len(a) != total
            or len(b) != total or a.keys() != b.keys()):
        raise ValueError("incomplete or unmatched ScreenSpot predictions")
    groups = {}
    for key, row in a.items():
        groups.setdefault(f"{row['split']} / {row['type']}", []).append(key)
    groups["Overall"] = list(a)
    lines = ["## ScreenSpot", "",
             f"Frozen protocol `{original['config']['protocol']}`. The shared Qwen assistant prefix "
             "specifies the output field. For pixel-based protocols, coordinates are mapped "
             "from the model's resized screenshot into the original image. "
             "A predicted point inside the labeled box, including its "
             "boundary, is correct. Malformed or out-of-range points count as errors.", "",
             "| Group | Samples | Native | Working set | Delta (pp) |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name, keys in groups.items():
        x = sum(a[key]["correct"] for key in keys) / len(keys) * 100
        y = sum(b[key]["correct"] for key in keys) / len(keys) * 100
        lines.append(f"| {name} | {len(keys)} | {x:.2f}% | {y:.2f}% | {y - x:+.2f} |")
    delta = y - x
    diff = [int(b[key]["correct"]) - int(a[key]["correct"]) for key in a]
    rng = Random(42)
    bootstrap = sorted(sum(rng.choices(diff, k=total)) * 100 / total for _ in range(2000))
    invalid = [sum(row["point"] is None for row in side.values()) for side in (a, b)]
    lines += ["", f"Paired 95% bootstrap interval for the difference "
              f"[{bootstrap[50]:+.2f}, {bootstrap[1949]:+.2f}] pp. "
              f"Invalid answers {invalid[0]}/{total} and {invalid[1]}/{total}. "
              "Standalone ScreenSpot inputs carry no history. The replay below measures the "
              "effect of removing history.", ""]
    return lines, delta


def summarize_replay(root, destination, dataset):
    data = load(root, "replay-test.json")
    cases = data["rows"]
    if data["config"]["split"] != "test" or set(cases) != set(data["config"]["sample_ids"]):
        raise ValueError("incomplete held-out GUI replay")
    lines = ["## GUI history replay", "",
             "ScreenSpot screenshots with deterministic synthetic histories. The current "
             "instruction references a ticket, and its linked older tool result names the "
             "required target. The selector receives record links. Box annotations are used "
             "only for offline scoring. All selection strategies use the same replay budget.",
             "", "| Path | Click accuracy | Evidence recall | Input p50 | TTFT p50 (ms) | Old screen dropped |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for mode in data["config"]["modes"]:
        rows = [options[mode] for options in cases.values()]
        clicks = sum(row["correct"] for row in rows)
        evidence = sum(row["evidence_kept"] for row in rows)
        dropped = sum(row["images_after"] < row["images_before"] for row in rows)
        lines.append(f"| {mode} | {clicks}/{len(rows)} | {evidence}/{len(rows)} | "
                     f"{median(row['retained_tokens'] for row in rows):.0f} | "
                     f"{median(row['ttft_ms'] for row in rows):.1f} | {dropped}/{len(rows)} |")
    candidates = list(cases.values())
    best = max(candidates, key=lambda case: (case["working_set"]["correct"],
               case["working_set"]["evidence_kept"] and not case["recent"]["evidence_kept"],
               case["working_set"]["images_after"] < case["working_set"]["images_before"]))
    row = best["working_set"]
    sample = next(item for item in load_dataset(dataset)[0] if item["id"] == row["sample"])
    replay_figure(destination / "gui-replay.png", sample, row["point"])
    demo = [
        "# GUI Agent replay", "",
        f"- ScreenSpot sample `{row['sample']}`.",
        f"- Current request “{row['query']}”",
        f"- The older approved record says “{row['expected_instruction']}”.",
        f"- Required records `{', '.join(row['required_records'])}`.",
        f"- Retained records `{', '.join(row['kept_records'])}`.",
        f"- Input tokens {row['original_tokens']} → {row['retained_tokens']}. "
        f"Screenshots {row['images_before']} → {row['images_after']}.",
        f"- Predicted click `{row['point']}`. Scored {'correct' if row['correct'] else 'incorrect'}.",
        f"- Recency-only keeps the required evidence: {best['recent']['evidence_kept']}.",
        "", "![Working-set replay with annotated click](gui-replay.png)", "",
    ]
    (destination / "gui-replay.md").write_text("\n".join(demo), encoding="utf-8")
    lines += ["", "[Annotated replay](gui-replay.md)", ""]
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("outputs/gui"))
    parser.add_argument("--dataset", type=Path, default=Path("data/screenspot"))
    parser.add_argument("--output", type=Path, default=Path("docs/gui-evaluation.md"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    performance, gates = summarize_perf(args.results, args.output.parent)
    accuracy, delta = summarize_screenspot(args.results)
    replay = summarize_replay(args.results, args.output.parent, args.dataset)
    lines = [
        "# GUI Agent working-set evaluation", "",
        "| Goal | Measured | Result |", "| --- | ---: | --- |",
        f"| 32K TTFT p50 reduction ≥35% | {gates['ttft']:.2f}% | {grade(gates['ttft'], 35)} |",
        f"| 32K occupied KV reduction ≥40% | {gates['kv']:.2f}% | {grade(gates['kv'], 40)} |",
        f"| ScreenSpot accuracy change ≥−1 pp | {delta:+.2f} pp | {grade(delta, -1)} |",
        "", "![32K TTFT and peak occupied KV](gui-performance.png)", "",
        *performance, *accuracy, *replay,
        "## Reproduction", "",
        "Run `bash scripts/reproduce_gui.sh` on AutoDL with the local model and ScreenSpot "
        "release already present. ScreenSpot comes from "
        "[SeeClick](https://github.com/njucckevin/SeeClick#gui-grounding-benchmark-screenspot), "
        "and the tool format adapts Qwen's "
        "[computer-use cookbook](https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/computer_use.ipynb). "
        "The original Step 4 visual-only evaluation is archived in "
        "`evaluation_report.md`. Raw results from this run are in `outputs/gui/`.", "",
    ]
    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
