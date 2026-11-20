import math

import pytest
import torch

from compressor.scoring import score_attention


def test_entropy_controls_budget():
    concentrated = torch.tensor([[[1., 0., 0., 0.]]])
    uniform = torch.ones(2, 3, 4)
    low, high = score_attention(concentrated), score_attention(uniform)
    assert (low.entropy, low.keep_count) == (0., 1)
    assert high.entropy == pytest.approx(1.)
    assert high.keep_count == 3
    torch.testing.assert_close(high.scores, torch.full((4,), .25))


def test_zero_rows_are_ignored_and_rows_normalized():
    weights = torch.tensor([[[4., 0.], [0., 2.], [0., 0.]]])
    result = score_attention(weights)
    torch.testing.assert_close(result.scores, torch.tensor([.5, .5]))
    assert result.entropy == 0.


@pytest.mark.parametrize("shape", [(0, 2, 4), (2, 0, 4), (1, 1, 0)])
def test_empty_evidence(shape):
    result = score_attention(torch.empty(shape))
    assert result.keep_count == shape[-1]


def test_zero_evidence_and_single_token():
    assert score_attention(torch.zeros(2, 3, 4)).keep_count == 4
    assert score_attention(torch.ones(2, 3, 1)).keep_count == 1


@pytest.mark.parametrize("ratios", [(0, .5), (.8, .2), (.1, 1.1), (math.nan, 1)])
def test_invalid_ratios(ratios):
    with pytest.raises(ValueError, match="ratios"):
        score_attention(torch.ones(1, 1, 4), *ratios)


@pytest.mark.parametrize("weights", [
    torch.ones(2, 4), torch.ones(1, 1, 4, dtype=torch.long),
    torch.tensor([[[-1.]]]), torch.tensor([[[math.nan]]]),
    torch.tensor([[[math.inf]]]),
])
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        score_attention(weights)


def test_large_finite_weights_and_no_grad():
    weights = torch.full((1, 1, 4), 1e38, requires_grad=True)
    result = score_attention(weights)
    assert result.entropy == pytest.approx(1.)
    assert not result.scores.requires_grad
