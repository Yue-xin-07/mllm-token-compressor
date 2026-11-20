"""Explicit, reversible hooks for vLLM 0.7.3 V0 single-request inference."""
from dataclasses import replace
import torch


class RunnerBridge:
    """Inject prepared embeddings/positions without changing cache metadata."""

    def __init__(self, runner, prepared):
        self.runner = runner
        self.model = runner.model
        self.prepared = prepared
        self.old_execute = runner.execute_model
        self.old_forward = self.model.forward
        self.inject_prefill = False
        self.trace = []

    def __enter__(self):
        self.runner.execute_model = self.execute
        self.model.forward = self.forward
        return self

    def __exit__(self, *exc):
        self.runner.execute_model = self.old_execute
        self.model.forward = self.old_forward
        self.inject_prefill = False
        self.prepared = None

    def forward(self, *args, **kwargs):
        if self.inject_prefill:
            kwargs["inputs_embeds"] = self.prepared.embeddings
            kwargs["input_ids"] = None
        return self.old_forward(*args, **kwargs)

    @torch.inference_mode()
    def execute(self, model_input, *args, **kwargs):
        metadata = model_input.attn_metadata
        prefill = metadata.num_prefills > 0
        length = len(self.prepared.token_ids)
        if prefill:
            if metadata.num_prefills != 1 or model_input.input_tokens.numel() != length:
                raise RuntimeError("only complete single-request prefill is supported")
            model_input = replace(model_input, input_positions=self.prepared.positions)
        self.inject_prefill = prefill
        self.trace.append({
            "prefill": prefill,
            "positions": model_input.input_positions.detach().cpu().tolist(),
            "slots": metadata.slot_mapping.detach().cpu().tolist(),
            "seq_lens": metadata.seq_lens_tensor.detach().cpu().tolist(),
            "blocks": metadata.block_tables.detach().cpu().tolist(),
        })
        try:
            return self.old_execute(model_input, *args, **kwargs)
        finally:
            self.inject_prefill = False
