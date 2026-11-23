"""Fused reductions for contiguous FP32 CUDA probabilities from the attention probe."""
import math

import torch
import triton
import triton.language as tl

from .scoring import AttentionBudget, _keep_count, score_attention


@triton.jit
def _score_tiles(attention, partial, stats, rows: tl.constexpr, columns: tl.constexpr,
                 row_tile: tl.constexpr, col_tile: tl.constexpr):
    tile = tl.program_id(0)
    row = tile * row_tile + tl.arange(0, row_tile)
    col = tl.arange(0, col_tile)
    values = tl.load(attention + row[:, None].to(tl.int64) * columns + col[None, :],
                     (row[:, None] < rows) & (col[None, :] < columns), 0)
    invalid = (values != values) | (values < 0) | (values == float("inf"))
    peak = tl.max(values, axis=1)
    active = peak > 0
    scaled = values / tl.where(active, peak, 1.0)[:, None]
    total = tl.sum(scaled, axis=1)
    prob = scaled / tl.where(active, total, 1.0)[:, None]
    terms = -prob * tl.log(tl.maximum(prob, 1.1754943508222875e-38))
    tl.store(partial + tile * columns + col, tl.sum(prob, axis=0), col < columns)
    tl.store(stats + 3 * tile, tl.sum(tl.sum(terms, axis=1), axis=0))
    tl.store(stats + 3 * tile + 1, tl.sum(active.to(tl.float32), axis=0))
    tl.store(stats + 3 * tile + 2, tl.max(tl.max(invalid.to(tl.int32), axis=1), axis=0))


def score_cuda_attention(attention, min_ratio, max_ratio):
    """Score validated probe output. The caller checks shape, layout and device."""
    heads, queries, columns = attention.shape
    rows = heads * queries
    col_tile = triton.next_power_of_2(columns)
    row_tile = min(32, 8192 // col_tile)
    tiles = triton.cdiv(rows, row_tile)
    partial = attention.new_empty((tiles, columns))
    stats = attention.new_empty((tiles, 3))
    _score_tiles[(tiles,)](attention, partial, stats, rows, columns, row_tile, col_tile)
    entropy_total, valid_rows, invalid_tiles = stats.sum(dim=0).tolist()
    if invalid_tiles:
        raise ValueError("attention weights must be finite and nonnegative")
    if not valid_rows:
        return AttentionBudget(attention.new_zeros(columns), 0.0, columns)
    entropy = min(1.0, max(0.0, entropy_total / (valid_rows * math.log(columns))))
    exact_count = columns * (min_ratio + (max_ratio - min_ratio) * entropy)
    # Keep PyTorch's rounding decision when reduction error could cross an integer.
    if min_ratio != max_ratio and abs(exact_count - round(exact_count)) <= columns * 2e-6:
        del partial, stats
        return score_attention(attention, min_ratio, max_ratio)
    scores = partial.sum(dim=0) / valid_rows
    return AttentionBudget(scores, entropy, _keep_count(columns, entropy, min_ratio, max_ratio))
