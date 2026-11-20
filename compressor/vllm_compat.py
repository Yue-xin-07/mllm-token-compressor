"""Compatibility with the installed torch 2.5 / flash-attn 2.8 pair."""
import torch


def install_vision_rotary_compat():
    """Use vLLM's own reference rotary only when wrap_triton is absent.

    Returns a restoration function; never edits installed package files.
    """
    if hasattr(torch.library, "wrap_triton"):
        return lambda: None
    from vllm.model_executor.models import qwen2_vl, qwen2_5_vl
    original = qwen2_vl.apply_rotary_pos_emb_vision
    second = qwen2_5_vl.apply_rotary_pos_emb_vision

    def compatible(t, freqs, use_flash_attn=False):
        return original(t, freqs, use_flash_attn=False)

    qwen2_vl.apply_rotary_pos_emb_vision = compatible
    qwen2_5_vl.apply_rotary_pos_emb_vision = compatible

    def restore():
        qwen2_vl.apply_rotary_pos_emb_vision = original
        qwen2_5_vl.apply_rotary_pos_emb_vision = second
    return restore
