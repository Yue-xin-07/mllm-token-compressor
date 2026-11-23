"""Request preparation, independent of the vLLM scheduler."""
from dataclasses import dataclass
import torch
from .attention_probe import visual_attention
from .scoring import score_probe_attention
from .token_pruner import TokenPruner


@dataclass(frozen=True)
class PreparedRequest:
    token_ids: list[int]
    embeddings: torch.Tensor
    positions: torch.Tensor
    kept_indices: torch.Tensor
    original_length: int
    next_position: int
    visual_before: int
    visual_after: int

    @property
    def rope_delta(self):
        return self.next_position - len(self.token_ids)


@torch.inference_mode()
def prepare_request(model, token_ids, positions, mm_kwargs, special_ids,
                    min_ratio=.25, max_ratio=.75):
    pruner = TokenPruner(min_ratio, max_ratio)
    ids = torch.tensor(token_ids, device=positions.device, dtype=torch.long)
    visual = (ids == model.config.image_token_id) | (
        ids == model.config.video_token_id)
    multimodal = model.get_multimodal_embeddings(**mm_kwargs) if mm_kwargs else None
    embeddings = model.get_input_embeddings(ids, multimodal)
    query_mask = ~visual
    for special in special_ids:
        query_mask &= ids != special
    if min_ratio == max_ratio == 1 or not visual.any():
        attention = embeddings.new_empty((1, 0, int(visual.sum())),
                                          dtype=torch.float32)
    else:
        layer = model.language_model.model.layers[0]
        attn = layer.self_attn
        normalized = layer.input_layernorm(embeddings)
        qkv, _ = attn.qkv_proj(normalized)
        q, k, _ = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], -1)
        q, k = attn.rotary_emb(positions, q, k)
        attention = visual_attention(
            q.reshape(-1, attn.num_heads, attn.head_dim),
            k.reshape(-1, attn.num_kv_heads, attn.head_dim),
            torch.where(query_mask)[0], torch.where(visual)[0])
    result = pruner._prune(
        embeddings, positions, visual, attention, score_probe_attention)
    kept = result.kept_indices
    return PreparedRequest(
        ids.index_select(0, kept).tolist(), result.hidden_states,
        result.position_ids, kept, len(token_ids), result.next_position,
        int(visual.sum()), result.visual_keep_count)
