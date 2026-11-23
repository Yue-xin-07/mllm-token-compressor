"""CUDA parity for the fused scorer used during request preparation."""
import pytest
import torch

from compressor.prepared import prepare_request
from compressor.scoring import score_attention, score_probe_attention

CUDA_TRITON = (torch.cuda.is_available()
               and torch.cuda.get_device_capability() >= (8, 0))
pytestmark = pytest.mark.skipif(not CUDA_TRITON, reason="requires CUDA sm80+")


@pytest.mark.parametrize("columns", [64, 65, 256, 1024, 2048])
def test_fused_score_matches_pytorch(columns):
    pytest.importorskip("triton")
    torch.manual_seed(9)
    logits = torch.randn(3, 19, columns, device="cuda")
    # Include zero probabilities, empty rows, and equal columns in a partial tile.
    logits[:, :, 1] = logits[:, :, 0]
    logits[:, :, 2] = -torch.inf
    attention = logits.softmax(dim=-1)
    attention[:, :3] = 0
    expected = score_attention(attention)
    actual = score_probe_attention(attention)
    torch.testing.assert_close(actual.scores, expected.scores, rtol=3e-6, atol=1e-8)
    assert actual.entropy == pytest.approx(expected.entropy, abs=5e-7)
    assert actual.keep_count == expected.keep_count
    assert actual.scores[0] == actual.scores[1]


@pytest.mark.parametrize("queries", [0, 19])
def test_no_evidence_retains_all_visual_tokens(queries):
    pytest.importorskip("triton")
    attention = torch.zeros(3, queries, 64, device="cuda")
    result = score_probe_attention(attention)
    assert result.keep_count == 64
    assert result.entropy == 0
    assert torch.count_nonzero(result.scores) == 0


def test_integer_budget_preserves_reference_rounding(monkeypatch):
    pytest.importorskip("triton")
    import compressor.triton_scoring as fused

    attention = torch.full((3, 19, 64), 1 / 64, device="cuda")
    expected = score_attention(attention)
    calls = []

    def reference(*args):
        calls.append(True)
        return score_attention(*args)

    monkeypatch.setattr(fused, "score_attention", reference)
    result = score_probe_attention(attention)
    assert calls == [True]
    assert result.keep_count == expected.keep_count
    torch.testing.assert_close(result.scores, expected.scores, rtol=0, atol=0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1])
def test_fused_score_rejects_invalid_weights(value):
    pytest.importorskip("triton")
    attention = torch.full((3, 19, 64), 1 / 64, device="cuda")
    attention[-1, -1, -1] = value
    with pytest.raises(ValueError, match="finite and nonnegative"):
        score_probe_attention(attention)


def test_noncontiguous_attention_uses_reference():
    attention = torch.rand(3, 64, 19, device="cuda").transpose(1, 2)
    expected = score_attention(attention)
    result = score_probe_attention(attention)
    assert result.keep_count == expected.keep_count
    assert result.entropy == expected.entropy
    torch.testing.assert_close(result.scores, expected.scores, rtol=0, atol=0)


def test_prepared_request_uses_fused_scorer_and_preserves_tie_order(monkeypatch):
    pytest.importorskip("triton")
    import compressor.prepared as prepared
    import compressor.triton_scoring as fused
    from test_prepared import model

    original = fused.score_cuda_attention
    calls = []

    def track(*args):
        calls.append(True)
        return original(*args)

    ids = [1, *([10] * 65), 3, 2]
    positions = torch.arange(len(ids), device="cuda").expand(3, -1)
    monkeypatch.setattr(prepared, "score_probe_attention", score_attention)
    expected = prepare_request(model(), ids, positions, {}, [1, 2], .5, .5)
    monkeypatch.setattr(prepared, "score_probe_attention", score_probe_attention)
    monkeypatch.setattr(fused, "score_cuda_attention", track)
    result = prepare_request(model(), ids, positions, {}, [1, 2], .5, .5)
    assert calls == [True]
    assert result.token_ids == expected.token_ids
    assert result.rope_delta == expected.rope_delta
    assert result.visual_after == 33
    torch.testing.assert_close(result.kept_indices, expected.kept_indices)
    torch.testing.assert_close(result.kept_indices[1:34], torch.arange(1, 34, device="cuda"))
    torch.testing.assert_close(result.positions, expected.positions)
    torch.testing.assert_close(result.embeddings, expected.embeddings)
