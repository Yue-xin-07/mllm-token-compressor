"""Measure a burst of requests through a vLLM V0 scheduler."""
from time import perf_counter
from uuid import uuid4


def run_workload(engine, prompts, submit, lengths, block_bytes, pool_blocks):
    if engine.has_unfinished_requests():
        raise RuntimeError("workload requires an idle engine")
    scheduler, = engine.scheduler
    free_blocks = scheduler.block_manager.get_num_free_gpu_blocks
    original_schedule = scheduler.schedule
    initial_free = free_blocks()
    peak = 0
    max_scheduled = 0
    first_token = {}
    results = {}
    ids = [str(uuid4()) for _ in prompts]

    def schedule(*args, **kwargs):
        nonlocal peak, max_scheduled
        scheduled = original_schedule(*args, **kwargs)
        metadata, outputs, _ = scheduled
        peak = max(peak, initial_free - free_blocks())
        max_scheduled = max(max_scheduled, len(metadata))
        if outputs.preempted:
            raise RuntimeError("workload exceeded cache capacity and triggered preemption")
        return scheduled

    scheduler.schedule = schedule
    try:
        start = perf_counter()
        for request_id, prompt in zip(ids, prompts):
            submit(request_id, prompt)
        while engine.has_unfinished_requests():
            outputs = engine.step()
            now = perf_counter()
            for result in outputs:
                if result.outputs and result.outputs[0].token_ids:
                    first_token.setdefault(result.request_id, now)
                if result.finished:
                    results[result.request_id] = result
        elapsed = perf_counter() - start
        rows = []
        for request_id in ids:
            result = results[request_id]
            retained = len(result.prompt_token_ids)
            original, visual, visual_retained = lengths.get(request_id, (retained, None, None))
            rows.append({
                "ttft_ms": (first_token[request_id] - start) * 1000,
                "output_tokens": len(result.outputs[0].token_ids),
                "original_tokens": original,
                "retained_tokens": retained,
                "visual_tokens": visual,
                "retained_visual_tokens": visual_retained,
            })
    finally:
        scheduler.schedule = original_schedule
        for request_id in ids:
            engine.abort_request(request_id)
        if initial_free != free_blocks():
            raise RuntimeError("workload did not return all allocated KV cache blocks")
    return [results[request_id] for request_id in ids], {
        "requests": rows,
        "elapsed_seconds": elapsed,
        "max_scheduled_requests": max_scheduled,
        "kv_peak_blocks": peak,
        "kv_peak_bytes": peak * block_bytes,
        "kv_pool_bytes": pool_blocks * block_bytes,
        "reusable_free_blocks": initial_free,
    }
