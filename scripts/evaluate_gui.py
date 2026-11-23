"""Run the full ScreenSpot evaluation with the structured Qwen click protocol."""
import argparse
from contextlib import closing
from hashlib import file_digest
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.benchmark_e2e import MODEL
from scripts.gui_protocol import IMAGE_PIXELS, PROTOCOLS, make_click_context, score_prediction
from scripts.screenspot_metrics import load_dataset


OUTPUT_LIMIT = {"tool": 18, "point": 18, "bbox": 28}


def balanced(samples, limit, offset=0):
    groups = {}
    for sample in samples:
        groups.setdefault((sample["split"], sample["type"]), []).append(sample)
    interleaved = [sample for index in range(max(map(len, groups.values())))
                   for group in groups.values() if index < len(group)
                   for sample in [group[index]]]
    return interleaved[offset:offset + limit]


def read_or_create(path, mode, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous["mode"] != mode or previous["config"] != config:
            raise ValueError(f"{path} belongs to a different protocol or backend")
        return previous
    return {"mode": mode, "config": config, "rows": {}}


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sampling_params(protocol="tool"):
    from vllm import SamplingParams
    return SamplingParams(temperature=0, max_tokens=OUTPUT_LIMIT[protocol], stop=["]"],
                          include_stop_str_in_output=True, skip_special_tokens=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("native", "working_set"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--protocol", choices=PROTOCOLS, default="point")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("batch size and optional sample limit must be positive")
    from compressor.vllm_engine import CompressedEngine
    from vllm import __version__ as vllm_version

    samples, hashes = load_dataset(args.dataset)
    total = len(samples)
    if args.limit is not None:
        samples = balanced(samples, args.limit)
    archive = args.dataset / "screenspot_imgs.tar"
    if archive.is_file():
        with archive.open("rb") as stream:
            hashes[archive.name] = file_digest(stream, "sha256").hexdigest()
    config = {"protocol": f"qwen-prefilled-{args.protocol}-v3", "dataset_sha256": hashes,
              "full_count": total, "count": len(samples), "batch_size": args.batch_size,
              "image_max_pixels": IMAGE_PIXELS, "max_output": OUTPUT_LIMIT[args.protocol],
              "model": args.model, "vllm": vllm_version}
    data = read_or_create(args.output, args.mode, config)
    pending = [sample for sample in samples if sample["id"] not in data["rows"]]
    if not pending:
        return
    sampling = sampling_params(args.protocol)
    with closing(CompressedEngine(args.model, 6144, max_num_seqs=args.batch_size,
                                  max_image_pixels=IMAGE_PIXELS)) as adapter:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            contexts = [make_click_context(sample, protocol=args.protocol) for sample in batch]
            if args.mode == "native":
                outputs, stats = adapter.generate_native_batch([item.render() for item in contexts], sampling)
            else:
                outputs, stats = adapter.generate_context_batch(contexts, sampling)
            for sample, context, result, row in zip(batch, contexts, outputs, stats["requests"],
                                                    strict=True):
                text = result.outputs[0].text
                point, correct = score_prediction(text, sample["bbox"], context.suffix.image.size, args.protocol)
                data["rows"][sample["id"]] = {
                    "text": text, "point": point, "correct": correct, "split": sample["split"],
                    "type": sample["type"], "original_tokens": row["original_tokens"],
                    "retained_tokens": row["retained_tokens"],
                }
            done = len(data["rows"])
            if (start // args.batch_size) % 5 == 0 or done == len(samples):
                save(args.output, data)
                print(f"{args.mode} ScreenSpot tool protocol: {done}/{len(samples)}", flush=True)


if __name__ == "__main__":
    main()
