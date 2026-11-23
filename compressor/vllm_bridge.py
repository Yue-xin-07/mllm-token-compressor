"""Explicit, reversible hooks for vLLM 0.7.3 V0 inference."""
from dataclasses import replace
import torch


class RunnerBridge:
    """Inject prepared embeddings/positions without changing cache metadata."""

    def __init__(self, runner, prepared=None, *, requests=None, trace=True):
        self.runner = runner
        self.model = runner.model
        self.prepared = prepared
        self.old_execute = runner.execute_model
        self.old_forward = self.model.forward
        self.inject_prefill = False
        self.trace = []
        self.trace_enabled = trace
        self.requests = requests
        self.prefill_batch = ()
        self.embeddings = None
        self.old_prepare = runner.prepare_model_input if requests is not None else None

    def __enter__(self):
        self.runner.execute_model = self.execute
        self.model.forward = self.forward
        if self.old_prepare is not None:
            self.runner.prepare_model_input = self.prepare_batch
        return self

    def __exit__(self, *exc):
        self.runner.execute_model = self.old_execute
        self.model.forward = self.old_forward
        self.inject_prefill = False
        self.prepared = None
        self.prefill_batch = ()
        self.embeddings = None
        if self.old_prepare is not None:
            self.runner.prepare_model_input = self.old_prepare

    def prepare_batch(self, seq_group_metadata_list, *args, **kwargs):
        prepared = []
        for group in seq_group_metadata_list:
            if group.is_prompt:
                seq, = group.seq_data.values()
                if seq.get_num_computed_tokens() or seq.get_output_len():
                    raise RuntimeError("compressed prefill must start from a fresh request")
                prepared.append(self.requests.pop(group.request_id))
        self.prefill_batch = tuple(prepared)
        return self.old_prepare(seq_group_metadata_list, *args, **kwargs)

    def forward(self, *args, **kwargs):
        if self.inject_prefill:
            kwargs["inputs_embeds"] = self.embeddings
            kwargs["input_ids"] = None
        return self.old_forward(*args, **kwargs)

    @torch.inference_mode()
    def execute(self, model_input, *args, **kwargs):
        metadata = model_input.attn_metadata
        prefill = metadata.num_prefills > 0
        if prefill:
            batch = self.prefill_batch if self.requests is not None else (self.prepared,)
            lengths = [len(request.token_ids) for request in batch]
            if metadata.num_prefills != len(batch) or model_input.input_tokens.numel() != sum(lengths):
                raise RuntimeError("only complete prefills without interleaved decode are supported")
            if len(batch) == 1:
                self.embeddings = batch[0].embeddings
                positions = batch[0].positions
            else:
                self.embeddings = torch.cat([request.embeddings for request in batch])
                positions = torch.cat([request.positions for request in batch], dim=-1)
            model_input = replace(model_input, input_positions=positions)
        self.inject_prefill = prefill
        if self.trace_enabled:
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
            self.embeddings = None
            self.prefill_batch = ()
