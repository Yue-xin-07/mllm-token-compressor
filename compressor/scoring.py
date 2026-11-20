"""FP32 attention scoring and row-entropy budgets (not spectral entropy)."""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class AttentionBudget:
    scores: torch.Tensor
    entropy: float
    keep_count: int


@torch.inference_mode()
def score_attention(
    attention: torch.Tensor, min_ratio: float = 0.25, max_ratio: float = 0.75
) -> AttentionBudget:
    """Score [heads, queries, visual_tokens] weights, ignoring zero rows.

    Weights must already exclude masked keys. Each nonzero row is normalized
    independently. Missing evidence conservatively retains all visual tokens.
    """
    if not (math.isfinite(min_ratio) and math.isfinite(max_ratio)
            and 0 < min_ratio <= max_ratio <= 1):
        raise ValueError("ratios must satisfy 0 < min_ratio <= max_ratio <= 1")
    if attention.ndim != 3 or not attention.is_floating_point():
        raise ValueError("attention must be a floating tensor [H,Q,V]")
    if not torch.isfinite(attention).all() or (attention < 0).any():
        raise ValueError("attention weights must be finite and nonnegative")
    visual_count = attention.shape[-1]
    scores = torch.zeros(visual_count, dtype=torch.float32, device=attention.device)
    if visual_count == 0:
        return AttentionBudget(scores, 0.0, 0)
    # Scaling first avoids overflow when summing large but finite FP32 weights.
    rows = attention.float().reshape(-1, visual_count)
    if not torch.isfinite(rows).all():
        raise ValueError("attention weights must be representable in FP32")
    rows = rows[rows.amax(dim=-1) > 0]
    if rows.shape[0] == 0:
        return AttentionBudget(scores, 0.0, visual_count)
    rows = rows / rows.amax(dim=-1, keepdim=True)
    probabilities = rows / rows.sum(dim=-1, keepdim=True)
    scores = probabilities.mean(dim=0)
    entropy = 0.0
    if visual_count > 1:
        terms = probabilities * probabilities.clamp_min(torch.finfo(torch.float32).tiny).log()
        entropy = float((-terms.sum(dim=-1).mean() / math.log(visual_count)).clamp(0, 1))
    ratio = min_ratio + (max_ratio - min_ratio) * entropy
    keep_count = min(visual_count, max(1, math.ceil(visual_count * ratio)))
    return AttentionBudget(scores, entropy, keep_count)
