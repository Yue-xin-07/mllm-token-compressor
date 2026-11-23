"""Deterministic input construction and GPU memory sampling for Step 4."""
from os import getpid
import re
from threading import Event, Thread
from uuid import uuid4

from PIL import Image


def sized_prompt(engine, target, image=None):
    prefix = "<|im_start|>user\n"
    suffix = "\nAnswer with a short sentence.<|im_end|>\n<|im_start|>assistant\n"
    if image:
        prefix += "<|vision_start|><|image_pad|><|vision_end|>"
        suffix = "\nDescribe the image." + suffix

    def make(repeats):
        prompt = {"prompt": prefix + " x" * repeats + suffix}
        if image:
            prompt["multi_modal_data"] = {"image": image}
        return prompt

    def length(prompt):
        processed = engine.input_processor(
            engine.input_preprocessor.preprocess(prompt, request_id=str(uuid4())))
        return len(processed["prompt_token_ids"])

    repeats = target - length(make(0))
    for _ in range(4):
        prompt = make(repeats)
        observed = length(prompt)
        if observed == target:
            return prompt
        repeats += target - observed
    raise ValueError(f"could not construct exactly {target} input tokens")


def workloads(adapter):
    image = Image.new("RGB", (448, 448), (220, 30, 30))
    engine = adapter.engine
    yield "single_32k", [sized_prompt(engine, 32768, image)], [32768]
    yield "mixed_2k_8k", [sized_prompt(engine, 2048), sized_prompt(engine, 8192, image)], [2048, 8192]


def as_context(adapter, prompt):
    from compressor.working_set import GUIContext, HistoryRecord, split_text

    value = prompt["prompt"]
    history = re.search(r"(?: x)+", value)
    if history is None:
        raise ValueError("expected the canonical synthetic context")
    prefix = value[:history.start()]
    suffix = value[history.end():]
    return GUIContext(HistoryRecord("system", prefix, prompt.get("multi_modal_data", {}).get("image")),
                      split_text(history.group(), adapter.engine.get_tokenizer(), "context-"),
                      HistoryRecord("query", suffix), suffix)


def measure(run, record_memory):
    if not record_memory:
        return run()[1]

    import torch
    from vllm.third_party import pynvml

    pynvml.nvmlInit()
    try:
        # This benchmark runs on the single-GPU AutoDL instance.
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        stop = Event()
        samples = []
        process_samples = []
        pid = getpid()
        sample_errors = []

        def sample():
            samples.append(pynvml.nvmlDeviceGetMemoryInfo(handle).used)
            try:
                processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            except (pynvml.NVMLError_NotSupported, pynvml.NVMLError_NoPermission):
                return
            process_samples.extend(
                process.usedGpuMemory for process in processes
                if process.pid == pid and process.usedGpuMemory is not None
            )

        def poll():
            while not stop.wait(.01):
                try:
                    sample()
                except pynvml.NVMLError as error:
                    sample_errors.append(error)
                    return

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        sample()
        worker = Thread(target=poll, daemon=True)
        worker.start()
        try:
            _, stats = run()
            torch.cuda.synchronize()
            sample()
        finally:
            stop.set()
            worker.join()
        if sample_errors:
            raise RuntimeError("NVML polling failed") from sample_errors[0]
        stats.update(
            peak_device_memory_bytes=max(samples),
            peak_process_memory_bytes=max(process_samples, default=None),
            peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        return stats
    finally:
        pynvml.nvmlShutdown()
