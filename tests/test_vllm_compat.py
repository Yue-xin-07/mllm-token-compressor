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


def test_image_profile_uses_declared_image_span(monkeypatch):
    from types import SimpleNamespace
    from vllm.inputs.registry import DummyData
    from vllm.multimodal.profiling import MultiModalProfiler
    from vllm.sequence import SequenceData
    from compressor.vllm_compat import install_image_profile_compat

    ids = [95, 95, 95, 95, 95, 0]
    data = DummyData(SequenceData.from_seqs(ids), None,
                     {"image": [{"offset": 0, "length": 3}]})
    original = lambda *_: data
    monkeypatch.setattr(MultiModalProfiler, "get_dummy_data", original)
    profiler = SimpleNamespace(processing_info=SimpleNamespace(
        get_hf_config=lambda: SimpleNamespace(image_token_id=95)))
    restore = install_image_profile_compat()
    try:
        adjusted = MultiModalProfiler.get_dummy_data(profiler, len(ids))
        assert list(adjusted.seq_data.get_prompt_token_ids()) == [95, 95, 95, 0, 0, 0]
        assert list(data.seq_data.get_prompt_token_ids()) == ids
        assert adjusted.multi_modal_placeholders is data.multi_modal_placeholders
    finally:
        restore()
    assert MultiModalProfiler.get_dummy_data is original
