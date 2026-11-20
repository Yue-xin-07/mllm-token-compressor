from dataclasses import dataclass
from types import SimpleNamespace
import pytest
import torch
from compressor.vllm_bridge import RunnerBridge
from compressor.prepared import PreparedRequest


@dataclass
class Input:
    attn_metadata: object
    input_tokens: torch.Tensor
    input_positions: torch.Tensor


def setup():
    prepared = PreparedRequest([1, 2], torch.randn(2, 4),
        torch.tensor([[0, 8], [0, 9], [0, 10]]), torch.tensor([0, 3]), 4, 11, 2, 0)
    calls = []
    def forward(**kw):
        calls.append(kw)
        return "ok"
    model = SimpleNamespace(forward=forward)
    def execute(inp):
        return model.forward(input_ids=inp.input_tokens, positions=inp.input_positions)
    runner = SimpleNamespace(model=model, execute_model=execute)
    meta = SimpleNamespace(num_prefills=1, slot_mapping=torch.tensor([16, 17]),
        seq_lens_tensor=torch.tensor([2]), block_tables=torch.tensor([[1]]))
    inp = Input(meta, torch.tensor([1, 2]), torch.zeros(3, 2, dtype=torch.long))
    return runner, prepared, calls, inp


def test_injection_restoration_and_cache_invariance():
    runner, prepared, calls, inp = setup()
    old_execute, old_forward = runner.execute_model, runner.model.forward
    with RunnerBridge(runner, prepared) as bridge:
        assert runner.execute_model(inp) == "ok"
        assert calls[-1]["inputs_embeds"] is prepared.embeddings
        assert calls[-1]["positions"] is prepared.positions
        assert calls[-1]["input_ids"] is None
        assert bridge.trace[0]["slots"] == [16, 17]
        inp.attn_metadata.num_prefills = 0
        runner.execute_model(inp)
        assert "inputs_embeds" not in calls[-1]
    assert runner.execute_model is old_execute
    assert runner.model.forward is old_forward
    assert bridge.prepared is None


def test_exception_restores_hooks():
    runner, prepared, _, inp = setup()
    original = runner.execute_model
    inp.attn_metadata.num_prefills = 2
    with pytest.raises(RuntimeError):
        with RunnerBridge(runner, prepared):
            runner.execute_model(inp)
    assert runner.execute_model is original
