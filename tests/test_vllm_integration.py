"""Opt in: MLLM_RUN_INTEGRATION=1 .venv/bin/python -m pytest ... -q -s."""
import json
import math
import os
from pathlib import Path
import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.getenv("MLLM_RUN_INTEGRATION") != "1", reason="explicit real-model opt in")


@pytest.fixture(scope="module")
def adapter():
    from compressor.vllm_engine import CompressedEngine
    engine = CompressedEngine("/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct", 8192)
    yield engine
    engine.close()


def params(tokens=40):
    from vllm import SamplingParams
    return SamplingParams(temperature=0, ignore_eos=True, max_tokens=tokens)


def prompt(repeats=8, images=1):
    from PIL import Image
    image = Image.new("RGB", (224, 224), (220, 30, 30))
    vision = "<|vision_start|><|image_pad|><|vision_end|>"
    return {"prompt": "<|im_start|>user\n" + vision * images
            + "Context: this is a test. " * repeats
            + "\nWhat is the main color in the image? Answer briefly."
            + "<|im_end|>\n<|im_start|>assistant\n",
            "multi_modal_data": {"image": image if images == 1 else [image] * images}}


def native(adapter, value, sampling):
    engine = adapter.engine
    engine.add_request("native", value, sampling)
    result = None
    try:
        while engine.has_unfinished_requests():
            result = engine.step()[0]
        return result
    finally:
        engine.abort_request("native")


def validate_stats(stats, generated):
    length, block = stats["retained_tokens"], stats["block_size"]
    assert stats["free_blocks_before"] == stats["free_blocks_after"]
    assert math.ceil(length / block) <= stats["allocated_blocks_peak"]
    assert stats["allocated_blocks_peak"] <= math.ceil((length + generated - 1) / block)
    trace = stats["trace"]
    assert len(trace) == generated and trace[0]["prefill"]
    for index, step in enumerate(trace[1:]):
        assert not step["prefill"]
        assert step["positions"] == [[stats["next_position"] + index]] * 3
        logical = length + index
        assert step["seq_lens"] == [logical + 1]
        physical_block = step["blocks"][0][logical // block]
        assert step["slots"] == [physical_block * block + logical % block]


def test_keep_all_matches_native(adapter):
    value, sampling = prompt(), params()
    baseline = native(adapter, value, sampling)
    result = adapter.generate(value, sampling, 1, 1)
    assert list(result.outputs[0].token_ids) == list(baseline.outputs[0].token_ids)
    assert adapter.last_stats["visual_before"] == adapter.last_stats["visual_after"]
    validate_stats(adapter.last_stats, 40)


def test_long_compressed_and_blocks(adapter):
    value = prompt(repeats=800)
    adapter.generate(value, params(), 1, 1)
    uncompressed_blocks = adapter.last_stats["allocated_blocks_peak"]
    result = adapter.generate(value, params())
    stats = adapter.last_stats
    assert stats["allocated_blocks_peak"] < uncompressed_blocks
    stats["uncompressed_blocks_peak"] = uncompressed_blocks
    assert stats["original_tokens"] >= 4096
    assert 0 < stats["visual_after"] < stats["visual_before"]
    assert stats["retained_tokens"] < stats["original_tokens"]
    assert result.finished and len(result.outputs[0].token_ids) == 40
    validate_stats(stats, 40)
    Path("/tmp/mllm-step2-acceptance.json").write_text(json.dumps({
        **stats, "text": result.outputs[0].text}, indent=2))


def test_text_matches_native(adapter):
    value = "<|im_start|>user\nSay OK.<|im_end|>\n<|im_start|>assistant\n"
    sampling = params(4)
    baseline = native(adapter, value, sampling)
    result = adapter.generate(value, sampling)
    assert list(result.outputs[0].token_ids) == list(baseline.outputs[0].token_ids)
    assert adapter.last_stats["visual_before"] == 0
    validate_stats(adapter.last_stats, 4)


def test_two_images(adapter):
    result = adapter.generate(prompt(images=2), params(4))
    assert result.finished
    assert adapter.last_stats["visual_before"] == 128
    validate_stats(adapter.last_stats, 4)


def test_exception_cleanup_and_recovery(adapter):
    free_before = adapter._free_blocks()
    original = adapter.model.forward

    def fail(*args, **kwargs):
        raise RuntimeError("injected forward failure")

    adapter.model.forward = fail
    try:
        with pytest.raises(RuntimeError, match="injected forward failure"):
            adapter.generate(prompt(), params(4))
        assert adapter.model.forward is fail
        assert adapter._free_blocks() == free_before
        assert not adapter.engine.has_unfinished_requests()
    finally:
        adapter.model.forward = original
    result = adapter.generate(prompt(), params(4))
    assert result.finished
    validate_stats(adapter.last_stats, 4)


def test_video(adapter):
    import numpy as np
    frames = np.zeros((2, 224, 224, 3), dtype=np.uint8)
    frames[..., 0] = 220
    value = {
        "prompt": "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>"
                  "Describe the main color.<|im_end|>\n<|im_start|>assistant\n",
        "multi_modal_data": {"video": frames},
    }
    output = adapter.generate(value, params(4))
    assert output.finished
    assert 0 < adapter.last_stats["visual_after"] < adapter.last_stats["visual_before"]
    validate_stats(adapter.last_stats, 4)
