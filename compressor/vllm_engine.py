"""Version-pinned offline adapter; original vLLM sources remain untouched."""
import os
import time
import uuid
import torch
from .prepared import prepare_request
from .vllm_bridge import RunnerBridge
from .vllm_workload import run_workload


class CompressedEngine:
    def __init__(self, model_path, max_model_len=8192, gpu_memory_utilization=.85,
                 *, max_num_seqs=1, max_image_pixels=None, cache_blocks=None, max_images=1):
        self.max_num_seqs = max_num_seqs
        self.max_image_pixels = max_image_pixels
        os.environ["VLLM_USE_V1"] = "0"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        import vllm
        from vllm import EngineArgs, LLMEngine
        if vllm.__version__ != "0.7.3":
            raise RuntimeError("adapter requires vLLM 0.7.3")
        args = EngineArgs(
            model=model_path, dtype="bfloat16", enforce_eager=True,
            max_model_len=max_model_len, max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_model_len,
            enable_prefix_caching=False, enable_chunked_prefill=False,
            num_scheduler_steps=1, swap_space=0, block_size=16,
            num_gpu_blocks_override=cache_blocks,
            gpu_memory_utilization=gpu_memory_utilization,
            distributed_executor_backend="uni", disable_async_output_proc=True,
            disable_mm_preprocessor_cache=max_image_pixels is not None,
            limit_mm_per_prompt=({"image": max_images, "video": 0} if max_image_pixels else
                                 {"image": 2, "video": 1}),
            mm_processor_kwargs={"max_pixels": max_image_pixels} if max_image_pixels else None)
        from .vllm_compat import install_image_profile_compat, install_vision_rotary_compat
        self.restore_compat = install_vision_rotary_compat()
        restore_profile = install_image_profile_compat() if max_image_pixels else None
        try:
            self.engine = LLMEngine.from_engine_args(args)
        except BaseException:
            self.restore_compat()
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            raise
        finally:
            if restore_profile is not None:
                restore_profile()
        self.runner = self.engine.model_executor.driver_worker.worker.model_runner
        self.model = self.runner.model
        self.last_stats = {}

    @torch.inference_mode()
    def prepare(self, prompt, request_id, min_ratio, max_ratio):
        from vllm.multimodal import MultiModalKwargs
        from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
        processed = self.engine.input_processor(
            self.engine.input_preprocessor.preprocess(prompt, request_id=request_id))
        ids = processed["prompt_token_ids"]
        if len(ids) > self.engine.model_config.max_model_len:
            raise ValueError("original input exceeds max_model_len")
        mm = processed.get("mm_kwargs", processed.get("multi_modal_data", {}))
        coordinate_mm = mm
        mm = MultiModalKwargs.as_kwargs(MultiModalKwargs.batch([mm]), device=self.runner.device) if mm else {}
        if mm:
            positions, _ = MRotaryEmbedding.get_input_positions(
                ids, hf_config=self.model.config,
                image_grid_thw=coordinate_mm.get("image_grid_thw"),
                video_grid_thw=coordinate_mm.get("video_grid_thw"),
                second_per_grid_ts=coordinate_mm.get("second_per_grid_ts"),
                context_len=0, seq_len=len(ids))
            positions = torch.tensor(positions, device=self.runner.device,
                                     dtype=torch.long)
        else:
            positions = torch.arange(len(ids), device=self.runner.device).expand(3, -1)
        tokenizer = self.engine.get_tokenizer()
        return prepare_request(self.model, ids, positions, mm,
                               tokenizer.all_special_ids, min_ratio, max_ratio)

    def _free_blocks(self):
        return self.engine.scheduler[0].block_manager.get_num_free_gpu_blocks()

    def generate(self, prompt, sampling_params, min_ratio=.25, max_ratio=.75, *, trace=True):
        from vllm.inputs import token_inputs
        if self.engine.has_unfinished_requests():
            raise RuntimeError("adapter requires an idle single-request engine")
        if (sampling_params.n != 1 or sampling_params.best_of not in (None, 1)
                or sampling_params.prompt_logprobs is not None):
            raise ValueError("only n=1, best_of=1, no prompt_logprobs supported")
        request_id = str(uuid.uuid4())
        free_before = self._free_blocks()
        prepared = None
        steps_trace = []
        min_free = free_before
        try:
            prepared = self.prepare(prompt, request_id, min_ratio, max_ratio)
            group = self.engine._add_processed_request(
                request_id, token_inputs(prepared.token_ids), sampling_params,
                time.time(), None, None)
            for seq in group.get_seqs():
                seq.data.mrope_position_delta = prepared.rope_delta
            output = None
            with RunnerBridge(self.runner, prepared, trace=trace) as bridge:
                while self.engine.has_unfinished_requests():
                    results = self.engine.step()
                    min_free = min(min_free, self._free_blocks())
                    for result in results:
                        if result.request_id == request_id:
                            output = result
                steps_trace = bridge.trace
            if output is None or not output.finished:
                raise RuntimeError("request did not finish")
            return output
        finally:
            self.engine.abort_request(request_id)
            self.last_stats = {
                "original_tokens": prepared.original_length if prepared else None,
                "retained_tokens": len(prepared.token_ids) if prepared else None,
                "visual_before": prepared.visual_before if prepared else None,
                "visual_after": prepared.visual_after if prepared else None,
                "next_position": prepared.next_position if prepared else None,
                "free_blocks_before": free_before,
                "free_blocks_after": self._free_blocks(),
                "allocated_blocks_peak": free_before - min_free,
                "block_size": self.engine.cache_config.block_size,
                "trace": steps_trace,
            }
            prepared = None

    def generate_batch(self, prompts, sampling_params, min_ratio=.25, max_ratio=.75):
        return self._run_batch(prompts, sampling_params, (min_ratio, max_ratio))

    def generate_native_batch(self, prompts, sampling_params):
        return self._run_batch(prompts, sampling_params, None)

    def generate_context_batch(self, contexts, sampling_params, *, budget=18432,
                               strategy="working_set"):
        from .context_submission import context_submitter
        planner = context_submitter(self, sampling_params, budget, strategy)
        return self._run_batch(contexts, sampling_params, None, planner=planner)

    def _run_batch(self, prompts, sampling, ratios, *, planner=None):
        from contextlib import nullcontext
        from vllm.inputs import token_inputs
        from vllm.sampling_params import RequestOutputKind

        if not 0 < len(prompts) <= self.max_num_seqs:
            raise ValueError("batch size must be between one and max_num_seqs")
        if (sampling.n != 1 or sampling.best_of not in (None, 1)
                or sampling.prompt_logprobs is not None):
            raise ValueError("only n=1, best_of=1, no prompt_logprobs supported")
        if sampling.output_kind != RequestOutputKind.CUMULATIVE:
            raise ValueError("batch measurement requires cumulative outputs")
        requests = {}
        lengths = {}
        selections = {}
        compressed = ratios is not None

        def submit(request_id, prompt):
            if planner is not None:
                sizes, selection = planner(request_id, prompt)
                if sizes is not None:
                    lengths[request_id] = sizes
                selections[request_id] = selection
                return
            if not compressed:
                self.engine.add_request(request_id, prompt, sampling)
                return
            prepared = self.prepare(prompt, request_id, *ratios)
            group = self.engine._add_processed_request(
                request_id, token_inputs(prepared.token_ids), sampling,
                time.time(), None, None)
            for seq in group.get_seqs():
                seq.data.mrope_position_delta = prepared.rope_delta
            requests[request_id] = prepared
            lengths[request_id] = (prepared.original_length, prepared.visual_before,
                                   prepared.visual_after)

        cache = self.engine.cache_config
        block_bytes = self.engine.model_executor.driver_worker.worker.get_cache_block_size_bytes()
        bridge = (RunnerBridge(self.runner, requests=requests, trace=False)
                  if compressed else nullcontext())
        with bridge:
            results, stats = run_workload(self.engine, prompts, submit, lengths, block_bytes,
                                          cache.num_gpu_blocks)
        if not compressed:
            vision_ids = {self.model.config.image_token_id, self.model.config.video_token_id}
            for result, row in zip(results, stats["requests"], strict=True):
                count = sum(token_id in vision_ids for token_id in result.prompt_token_ids)
                if row["visual_tokens"] is None:
                    row.update(visual_tokens=count, retained_visual_tokens=count)
                if planner is not None:
                    row.update(selections[result.request_id])
        return results, stats

    def close(self):
        """Release the sole engine and restore process-local compatibility."""
        if self.engine is None:
            return
        for scheduler in self.engine.scheduler:
            for group in list(scheduler.waiting) + list(scheduler.running):
                self.engine.abort_request(group.request_id)
        self.engine.model_executor.shutdown()
        self.restore_compat()
        self.runner = self.model = self.engine = None
        import gc
        gc.collect()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        torch.cuda.empty_cache()
