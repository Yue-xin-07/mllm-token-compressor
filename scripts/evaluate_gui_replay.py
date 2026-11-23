"""Evaluate the full and ablated selectors on paired GUI history replays."""
import argparse
from contextlib import closing
from pathlib import Path
from statistics import median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.benchmark_e2e import MODEL
from scripts.evaluate_gui import IMAGE_PIXELS, balanced, read_or_create, sampling_params, save
from scripts.gui_protocol import PROTOCOLS, score_prediction
from scripts.gui_replay import build_replay
from scripts.screenspot_metrics import load_dataset


MODES = ("native", "visual", "recent", "bm25", "working_set", "oracle")
REPLAY_BUDGET = 2048


def evaluate(adapter, mode, context, evidence, sample, sampling, protocol):
    prompt = context.render()
    if mode == "native":
        output, stats = adapter.generate_native_batch([prompt], sampling)
    elif mode == "visual":
        output, stats = adapter.generate_batch([prompt], sampling)
    else:
        output, stats = adapter.generate_context_batch([context], sampling, budget=REPLAY_BUDGET,
                                                       strategy=mode)
    row, = stats["requests"]
    retained = row.get("kept_records")
    if retained is None:
        retained = [record.id for record in context.history]
    text = output[0].outputs[0].text
    point, correct = score_prediction(text, sample["bbox"], context.suffix.image.size, protocol)
    return {
        "text": text, "point": point, "correct": correct,
        "evidence_kept": set(evidence).issubset(retained),
        "required_records": evidence, "kept_records": retained,
        "expected_instruction": sample["instruction"],
        "query": context.query, "sample": sample["id"], "split": sample["split"],
        "image": sample["image"].name, "ttft_ms": row["ttft_ms"],
        "original_tokens": row["original_tokens"], "retained_tokens": row["retained_tokens"],
        "images_before": row.get("images_before", 2), "images_after": row.get("images_after", 2),
        "selection_ms": row.get("selection_ms", 0.0), "fallback": row.get("fallback"),
        "kv_peak_bytes": stats["kv_peak_bytes"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--protocol", choices=PROTOCOLS, default="point")
    args = parser.parse_args()
    n = args.limit if args.limit is not None else (24 if args.split == "dev" else 72)
    if n < 2:
        parser.error("at least two samples are needed for replay")

    from compressor.vllm_engine import CompressedEngine
    from vllm import __version__ as vllm_version
    samples, hashes = load_dataset(args.dataset)
    # Assign every screenshot to exactly one split, even when annotated multiple times.
    development = balanced(samples, 48)
    dev_screens = {row["image"] for row in development}
    if args.split == "dev":
        samples = development
    else:
        samples = [sample for sample in samples if sample["image"] not in dev_screens]
    selected = balanced(samples, n)
    config = {"protocol": f"gui-replay-{args.protocol}-v3", "dataset_sha256": hashes,
              "split": args.split, "sample_ids": [row["id"] for row in selected],
              "budget": REPLAY_BUDGET, "image_max_pixels": IMAGE_PIXELS,
              "model": args.model, "vllm": vllm_version, "modes": list(MODES)}
    data = read_or_create(args.output, "replay", config)
    if len(data["rows"]) == len(selected):
        return
    sampling = sampling_params(args.protocol)
    with closing(CompressedEngine(args.model, 8192, max_num_seqs=1, max_images=2,
                                  max_image_pixels=IMAGE_PIXELS, cache_blocks=768)) as adapter:
        for index, sample in enumerate(selected):
            if sample["id"] in data["rows"]:
                continue
            other = next(selected[(index + advance) % len(selected)] for advance in range(1, len(selected))
                         if selected[(index + advance) % len(selected)]["image"] != sample["image"])
            context, evidence = build_replay(sample, other["image"], args.protocol)
            modes = MODES[index % len(MODES):] + MODES[:index % len(MODES)]
            results = {mode: evaluate(adapter, mode, context, evidence, sample, sampling, args.protocol)
                       for mode in modes}
            data["rows"][sample["id"]] = results
            save(args.output, data)
            print(f"GUI replay {args.split}: {len(data['rows'])}/{len(selected)}", flush=True)
    for mode in MODES:
        rows = [sample[mode] for sample in data["rows"].values()]
        print(f"{mode}: click {sum(row['correct'] for row in rows)}/{len(rows)}, "
              f"evidence {sum(row['evidence_kept'] for row in rows)}/{len(rows)}, "
              f"TTFT {median(row['ttft_ms'] for row in rows):.2f} ms", flush=True)


if __name__ == "__main__":
    main()
