"""Run the original ScreenSpot test set on one backend, with resumable output."""
import argparse
from hashlib import file_digest
from contextlib import closing
import json
from pathlib import Path
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_e2e import MODEL
from compressor.vllm_engine import CompressedEngine
from screenspot_metrics import click_result, load_dataset


MAX_IMAGE_PIXELS = 1024 * 28 * 28
MAX_OUTPUT = 24
BATCH_SIZE = 2


def make_request(sample):
    with Image.open(sample["image"]) as source:
        image = source.convert("RGB")
    prompt = (
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        "Locate the center of the UI element described by this instruction. "
        "Return only the point in (x, y) format. Each coordinate ranges from 0 to 1000, "
        "with (0, 0) at the top left and (1000, 1000) at the bottom right.\n"
        f"Instruction: {sample['instruction']}<|im_end|>\n<|im_start|>assistant\n"
    )
    return {"prompt": prompt, "multi_modal_data": {"image": image}}


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("native", "compressed"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--limit", type=int, help="only run the first N examples for a smoke test")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    from vllm import SamplingParams, __version__ as vllm_version

    samples, hashes = load_dataset(args.dataset)
    full_count = len(samples)
    samples = samples[:args.limit]
    archive_hash = None
    archive = args.dataset / "screenspot_imgs.tar"
    if archive.is_file():
        with archive.open("rb") as stream:
            archive_hash = file_digest(stream, "sha256").hexdigest()
    config = {"annotations_sha256": hashes, "archive_sha256": archive_hash, "full_count": full_count, "count": len(samples),
              "image_max_pixels": MAX_IMAGE_PIXELS, "generation_max_tokens": MAX_OUTPUT,
              "model": args.model, "vllm": vllm_version, "prompt_version": 1, "stop_at_first_closing_parenthesis": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = {"mode": args.mode, "config": config, "rows": {}}
    if args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        if previous["mode"] != args.mode or previous["config"] != config:
            parser.error("existing output has a different configuration")
        data = previous
    pending = [sample for sample in samples if sample["id"] not in data["rows"]]
    if not pending:
        return
    sampling = SamplingParams(temperature=0, max_tokens=MAX_OUTPUT, stop=[")"],
                              include_stop_str_in_output=True)
    with closing(CompressedEngine(args.model, max_model_len=4096, max_num_seqs=BATCH_SIZE,
                                  max_image_pixels=MAX_IMAGE_PIXELS)) as adapter:
        run = adapter.generate_native_batch if args.mode == "native" else adapter.generate_batch
        for start in range(0, len(pending), BATCH_SIZE):
            batch = pending[start:start + BATCH_SIZE]
            prompts = [make_request(sample) for sample in batch]
            outputs, _ = run(prompts, sampling)
            for sample, request, output in zip(batch, prompts, outputs, strict=True):
                text = output.outputs[0].text
                point, correct = click_result(text, sample["bbox"],
                                              request["multi_modal_data"]["image"].size)
                data["rows"][sample["id"]] = {"text": text, "point": point, "correct": correct,
                                              "split": sample["split"], "type": sample["type"]}
            count = len(data["rows"])
            if start % 20 == 0 or count == len(samples):
                save(args.output, data)
                print(f"{args.mode} ScreenSpot: {count}/{len(samples)}", flush=True)


if __name__ == "__main__":
    main()
