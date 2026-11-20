"""Offline correctness harness; outputs include physical cache trace."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--text-repeats", type=int, default=800)
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--keep-all", action="store_true")
    parser.add_argument("--output", default="/tmp/mllm-step2-result.json")
    args = parser.parse_args()
    from PIL import Image
    from vllm import SamplingParams
    from compressor.vllm_engine import CompressedEngine
    adapter = CompressedEngine(args.model, args.max_model_len)
    image = Image.new("RGB", (224, 224), (220, 30, 30))
    prompt = {
        "prompt": "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                  + "Context: this is a test. " * args.text_repeats
                  + "\nWhat is the main color in the image? Answer briefly."
                  + "<|im_end|>\n<|im_start|>assistant\n",
        "multi_modal_data": {"image": image},
    }
    params = SamplingParams(temperature=0, max_tokens=args.max_tokens,
                            ignore_eos=True)
    ratio = 1. if args.keep_all else .25
    try:
        output = adapter.generate(prompt, params, ratio, 1. if args.keep_all else .75)
        stats = adapter.last_stats
    finally:
        adapter.close()
    stats["text"] = output.outputs[0].text
    stats["output_token_ids"] = list(output.outputs[0].token_ids)
    Path(args.output).write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k != "trace"}, indent=2))


if __name__ == "__main__":
    main()
