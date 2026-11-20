"""Single unpadded sequence pruning; model attention extraction is upstream."""
from dataclasses import dataclass
import math

import torch

from .scoring import score_attention


@dataclass(frozen=True)
class PruningResult:
    hidden_states: torch.Tensor
    position_ids: torch.Tensor
    kept_indices: torch.Tensor
    visual_keep_count: int
    entropy: float
    next_position: int


class TokenPruner:
    """Keep all text/special tokens and select visual tokens deterministically.

    visual_mask must mark only image/video content tokens, never delimiters.
    attention [H,Q,V] columns follow the original order of those visual tokens.
    Inputs are prefill tensors without padding; attention has already been masked.
    M-RoPE coordinates are gathered unchanged. next_position is the original
    max coordinate + 1, NOT a KV cache offset or Transformers rope_delta.
    No input tensor is modified and no per-request tensor is stored on the object.
    """

    def __init__(self, min_ratio: float = .25, max_ratio: float = .75):
        if not (math.isfinite(min_ratio) and math.isfinite(max_ratio)
                and 0 < min_ratio <= max_ratio <= 1):
            raise ValueError("ratios must satisfy 0 < min_ratio <= max_ratio <= 1")
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    @torch.inference_mode()
    def __call__(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        visual_mask: torch.Tensor,
        attention: torch.Tensor,
    ) -> PruningResult:
        self._validate(hidden_states, position_ids, visual_mask, attention)
        budget = score_attention(attention, self.min_ratio, self.max_ratio)
        visual_indices = torch.where(visual_mask)[0]
        keep_mask = ~visual_mask
        # stable sorting breaks ties by original index, including CUDA.
        selected = torch.argsort(budget.scores, descending=True, stable=True)
        keep_mask[visual_indices[selected[:budget.keep_count]]] = True
        kept_indices = torch.where(keep_mask)[0]
        next_position = int(position_ids.max()) + 1 if position_ids.numel() else 0
        return PruningResult(
            hidden_states=hidden_states.index_select(0, kept_indices),
            position_ids=position_ids.index_select(1, kept_indices),
            kept_indices=kept_indices,
            visual_keep_count=budget.keep_count,
            entropy=budget.entropy,
            next_position=next_position,
        )

    @staticmethod
    def _validate(hidden, positions, mask, attention):
        if hidden.ndim != 2 or not hidden.is_floating_point():
            raise ValueError("hidden_states must be floating [S,D]")
        length = hidden.shape[0]
        if positions.shape != (3, length) or positions.dtype != torch.long:
            raise ValueError("position_ids must be int64 [3,S]")
        if mask.shape != (length,) or mask.dtype != torch.bool:
            raise ValueError("visual_mask must be bool [S]")
        if any(t.device != hidden.device for t in (positions, mask, attention)):
            raise ValueError("all tensors must be on the same device")
        if attention.ndim != 3 or attention.shape[-1] != int(mask.sum()):
            raise ValueError("attention must be [H,Q,V], V equal to visual_mask count")
        if (positions < 0).any():
            raise ValueError("position_ids must be nonnegative")
