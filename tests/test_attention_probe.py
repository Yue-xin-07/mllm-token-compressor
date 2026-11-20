import pytest
import torch
from compressor.attention_probe import visual_attention
from compressor.token_pruner import TokenPruner


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("chunk", [1, 3, 64])
def test_dense_reference(kv_heads, chunk):
    torch.manual_seed(42)
    q, k = torch.randn(9, 4, 8), torch.randn(9, kv_heads, 8)
    qi, vi = torch.tensor([0, 3, 7, 8]), torch.tensor([1, 2, 5, 6])
    logits = torch.einsum("shd,thd->hst", q, k.repeat_interleave(4 // kv_heads, 1))
    logits /= 8 ** .5
    logits.masked_fill_(torch.triu(torch.ones(9, 9, dtype=torch.bool), 1), -torch.inf)
    full = logits.softmax(-1)[:, qi][:, :, vi]
    expected = full / full.sum(-1, keepdim=True).clamp_min(1e-30)
    actual = visual_attention(q, k, qi, vi, chunk)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert torch.isfinite(actual).all()


def test_no_evidence():
    q, k = torch.randn(4, 2, 8), torch.randn(4, 1, 8)
    mask = torch.tensor([False, True, True, False])
    empty = torch.empty(0, dtype=torch.long)
    attn = visual_attention(q, k, empty, torch.where(mask)[0])
    result = TokenPruner()(torch.randn(4, 8),
                           torch.arange(4).expand(3, -1), mask, attn)
    assert result.kept_indices.tolist() == [0, 1, 2, 3]
    assert visual_attention(q, k, torch.tensor([3]), empty).shape == (2, 1, 0)


@pytest.mark.parametrize("bad", [torch.tensor([-1]), torch.tensor([4]),
                                torch.tensor([2, 1]), torch.tensor([1, 1])])
def test_invalid_indices(bad):
    with pytest.raises(ValueError):
        visual_attention(torch.randn(4, 2, 8), torch.randn(4, 1, 8),
                         bad, torch.tensor([1]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda(dtype):
    q, k = torch.randn(12, 4, 8).to(dtype), torch.randn(12, 2, 8).to(dtype)
    qi, vi = torch.tensor([0, 9, 11]), torch.tensor([1, 4, 8])
    expected = visual_attention(q, k, qi, vi)
    actual = visual_attention(q.cuda(), k.cuda(), qi.cuda(), vi.cuda())
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-5, rtol=1e-4)
