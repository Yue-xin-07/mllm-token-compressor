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



def install_image_profile_compat():
    """Use the reported image span as the source of truth for V0 dummy tokens."""
    from vllm.multimodal.profiling import MultiModalProfiler
    from vllm.sequence import SequenceData

    original = MultiModalProfiler.get_dummy_data

    def get_dummy_data(profiler, seq_len, is_encoder_data=False):
        dummy = original(profiler, seq_len, is_encoder_data)
        spans = (dummy.multi_modal_placeholders or {}).get("image", ())
        if not spans:
            return dummy
        image_id = profiler.processing_info.get_hf_config().image_token_id
        ids = list(dummy.seq_data.get_prompt_token_ids())
        offsets = set()
        for span in spans:
            offsets.update(range(span["offset"], span["offset"] + span["length"]))
        if any(ids[index] != image_id for index in offsets):
            raise RuntimeError("invalid image span in vLLM profiling prompt")
        ids = [0 if token == image_id and index not in offsets else token
               for index, token in enumerate(ids)]
        return dummy._replace(seq_data=SequenceData.from_seqs(ids))

    MultiModalProfiler.get_dummy_data = get_dummy_data

    def restore():
        MultiModalProfiler.get_dummy_data = original

    return restore
