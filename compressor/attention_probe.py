"""Chunked causal GQA attention over visual keys."""
import math
import torch


@torch.inference_mode()
def visual_attention(query, key, query_indices, visual_indices, chunk_size=64):
    """Q/K [S,H,D] already carry M-RoPE; output FP32 [H,Q,V]."""
    if query.ndim != 3 or key.ndim != 3:
        raise ValueError("query/key must be [S,H,D]")
    if (query.shape[0] != key.shape[0] or query.shape[2] != key.shape[2]
            or key.shape[1] == 0 or query.shape[1] % key.shape[1]):
        raise ValueError("incompatible GQA dimensions")
    if not query.is_floating_point() or not key.is_floating_point():
        raise ValueError("query/key must be floating")
    if query.device != key.device or query.dtype != key.dtype:
        raise ValueError("query/key device and dtype must agree")
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for indices in (query_indices, visual_indices):
        if (indices.ndim != 1 or indices.dtype != torch.long
                or indices.device != query.device):
            raise ValueError("indices must be device-local int64 vectors")
        if indices.numel() and ((indices < 0).any()
                                or (indices >= query.shape[0]).any()):
            raise ValueError("index out of range")
        if indices.numel() > 1 and not (indices[1:] > indices[:-1]).all():
            raise ValueError("indices must be strictly increasing")
    if not torch.isfinite(query).all() or not torch.isfinite(key).all():
        raise ValueError("non-finite Q/K")
    heads, dim = query.shape[1:]
    result = torch.zeros((heads, query_indices.numel(), visual_indices.numel()),
                         device=query.device, dtype=torch.float32)
    if not result.numel():
        return result
    keys = key.index_select(0, visual_indices).float().transpose(0, 1)
    keys = keys.repeat_interleave(heads // key.shape[1], dim=0)
    for start in range(0, query_indices.numel(), chunk_size):
        indices = query_indices[start:start + chunk_size]
        queries = query.index_select(0, indices).float().transpose(0, 1)
        logits = torch.matmul(queries, keys.transpose(1, 2)) / math.sqrt(dim)
        valid = visual_indices[None, :] <= indices[:, None]
        logits.masked_fill_(~valid[None], -torch.inf)
        logits = torch.where(valid.any(-1)[None, :, None], logits, 0)
        result[:, start:start + indices.numel()] = (
            logits.softmax(-1).masked_fill(~valid[None], 0))
    return result
