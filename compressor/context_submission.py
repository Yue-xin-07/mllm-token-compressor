"""Translate selected GUI history into a native Qwen/vLLM submission."""
from time import perf_counter, time

from .working_set import DEFAULT_BUDGET, GUIContext, select_history


def context_submitter(adapter, sampling, budget=DEFAULT_BUDGET, strategy="working_set"):
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    engine = adapter.engine
    tokenizer = engine.get_tokenizer()
    max_pixels = adapter.max_image_pixels or 12845056
    vision_ids = {adapter.model.config.image_token_id, adapter.model.config.video_token_id}

    def media_tokens(image):
        height, width = smart_resize(image.height, image.width, factor=28,
                                     min_pixels=3136, max_pixels=max_pixels)
        return height * width // (28 * 28)

    def submit(request_id, context):
        start = perf_counter()
        if not isinstance(context, GUIContext):
            engine.add_request(request_id, context, sampling)
            return None, {"fallback": "unsupported_context", "kept_records": None,
                          "selection_ms": 0.0}
        original = context.render()
        records = (context.prefix, *context.history, context.suffix)
        image_lengths = {id(record): media_tokens(record.image)
                         for record in records if record.image is not None}

        def text_tokens(text):
            return len(tokenizer.encode(text, add_special_tokens=False))

        def cost(record):
            virtual = image_lengths.get(id(record), 0)
            return text_tokens(record.text) + (virtual - 1 if virtual else 0)

        visual_before = sum(image_lengths.values())
        original_length = text_tokens(original["prompt"]) + visual_before - len(image_lengths)
        selection = select_history(context, cost, budget, strategy)
        selected = len(selection.history) != len(context.history)
        selection_ms = (perf_counter() - start) * 1000
        stats = {"fallback": selection.fallback, "selection_ms": selection_ms,
                 "kept_records": selection.kept, "history_records": len(context.history),
                 "images_before": len(image_lengths), "images_after": len(image_lengths)}
        if not selected:
            engine.add_request(request_id, original, sampling)
            return None, stats

        prompt = context.render(selection.history)
        arrival = time()
        processed = engine.input_processor(
            engine.input_preprocessor.preprocess(prompt, request_id=request_id))
        ids = processed["prompt_token_ids"]
        if len(ids) > budget:
            stats.update(fallback="exact_budget_exceeded",
                         kept_records=[record.id for record in context.history])
            engine.add_request(request_id, original, sampling, arrival_time=arrival)
            return None, stats
        engine._add_processed_request(request_id, processed, sampling, arrival, None, None)
        stats["images_after"] = sum(record.image is not None for record in
                                    (context.prefix, *selection.history, context.suffix))
        visual_after = sum(token in vision_ids for token in ids)
        return (original_length, visual_before, visual_after), stats

    return submit
