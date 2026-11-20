import torch
from compressor.vllm_compat import install_vision_rotary_compat


def test_vision_rotary_reference_and_restore():
    from vllm.model_executor.models import qwen2_vl, qwen2_5_vl
    original = qwen2_vl.apply_rotary_pos_emb_vision
    second = qwen2_5_vl.apply_rotary_pos_emb_vision
    t, freqs = torch.randn(1, 5, 2, 8), torch.randn(5, 4)
    expected = original(t, freqs, use_flash_attn=False)
    restore = install_vision_rotary_compat()
    try:
        actual = qwen2_5_vl.apply_rotary_pos_emb_vision(t, freqs, use_flash_attn=True)
        torch.testing.assert_close(actual, expected)
    finally:
        restore()
    assert qwen2_vl.apply_rotary_pos_emb_vision is original
    assert qwen2_5_vl.apply_rotary_pos_emb_vision is second
