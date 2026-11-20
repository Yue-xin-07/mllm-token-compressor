from types import SimpleNamespace
import pytest
import torch
from compressor.prepared import prepare_request


class Projection:
    def __call__(self, x):
        return torch.cat([x, x, x], -1), None


def model():
    attn = SimpleNamespace(qkv_proj=Projection(), q_size=4, kv_size=4,
        num_heads=1, num_kv_heads=1, head_dim=4,
        rotary_emb=lambda p, q, k: (q, k))
    layer = SimpleNamespace(input_layernorm=lambda x: x, self_attn=attn)
    return SimpleNamespace(
        config=SimpleNamespace(image_token_id=10, video_token_id=11),
        get_multimodal_embeddings=lambda **kw: None,
        get_input_embeddings=lambda ids, mm: ids.float()[:, None].expand(-1, 4),
        language_model=SimpleNamespace(model=SimpleNamespace(layers=[layer])))


def test_alignment_and_decode_delta():
    ids = [1, 10, 10, 10, 10, 3, 2]
    positions = torch.tensor([[0, 1, 1, 1, 1, 5, 6],
                              [0, 1, 2, 3, 4, 5, 6],
                              [0, 1, 2, 3, 4, 5, 6]])
    result = prepare_request(model(), ids, positions, {}, [1, 2], .5, .5)
    assert result.visual_before == 4 and result.visual_after == 2
    assert result.token_ids == [1, 10, 10, 3, 2]
    torch.testing.assert_close(result.positions, positions[:, result.kept_indices])
    assert result.next_position == 7
    assert result.rope_delta == 2
    assert result.embeddings[:, 0].tolist() == result.token_ids


def test_keep_all_and_no_text_evidence():
    ids = [1, 10, 11, 2]
    positions = torch.arange(4).expand(3, -1)
    for ratio in [.5, 1.]:
        result = prepare_request(model(), ids, positions, {}, [1, 2], ratio, ratio)
        assert result.token_ids == ids
        assert result.rope_delta == 0


def test_text_only():
    result = prepare_request(model(), [1, 3, 2],
                             torch.arange(3).expand(3, -1), {}, [1, 2])
    assert result.visual_before == result.visual_after == 0
    assert result.token_ids == [1, 3, 2]


@pytest.mark.parametrize("ratios", [(.9, .2), (0., .5)])
def test_invalid_budget_before_model_access(ratios):
    with pytest.raises(ValueError):
        prepare_request(None, [], torch.empty(3, 0, dtype=torch.long),
                        {}, [], *ratios)
