from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from compressor import vllm_workload


@pytest.mark.parametrize("preempted", [0, 1])
def test_timing_and_kv_peak_before_release(monkeypatch, preempted):
    now = iter([1., 1.5, 2., 2.1])
    monkeypatch.setattr(vllm_workload, "perf_counter", lambda: next(now))
    active = {}
    free = 10
    steps = 0

    def submit(request_id, prompt):
        active[request_id] = prompt

    def schedule():
        nonlocal free
        free = 6
        return list(active), SimpleNamespace(preempted=preempted), False

    scheduler = SimpleNamespace(schedule=schedule,
                               block_manager=SimpleNamespace(get_num_free_gpu_blocks=lambda: free))

    def step():
        nonlocal steps, free
        scheduler.schedule()
        steps += 1
        finished = steps == 2
        rows = [SimpleNamespace(request_id=key, finished=finished, prompt_token_ids=[1, 2],
                               outputs=[SimpleNamespace(token_ids=[1] * steps)]) for key in active]
        if finished:
            active.clear()
            free = 10
        return rows

    def abort(request_id):
        nonlocal free
        active.pop(request_id, None)
        if not active:
            free = 10

    cleanup = Mock(side_effect=abort)
    engine = SimpleNamespace(scheduler=[scheduler], step=step, abort_request=cleanup,
                             has_unfinished_requests=lambda: bool(active))
    run = lambda: vllm_workload.run_workload(engine, ["text", "image"], submit, {}, 16, 10)
    if preempted:
        with pytest.raises(RuntimeError, match="preemption"):
            run()
    else:
        _, stats = run()
        assert [row["ttft_ms"] for row in stats["requests"]] == [500, 500]
        assert [row["output_tokens"] for row in stats["requests"]] == [2, 2]
        assert stats["kv_peak_bytes"] == 64
        assert stats["kv_pool_bytes"] == 160
        assert stats["max_scheduled_requests"] == 2
    assert cleanup.call_count == 2
    assert not active
    assert scheduler.schedule is schedule
