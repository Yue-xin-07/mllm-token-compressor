"""History selection before vLLM registers a multi-modal request."""
from collections import Counter
from dataclasses import dataclass
from math import log
import re
from typing import Callable

from PIL import Image


IMAGE = "<|vision_start|><|image_pad|><|vision_end|>"
DEFAULT_BUDGET = 18432
STRATEGIES = ("working_set", "recent", "bm25", "oracle")
WORDS = re.compile(r"[\u3400-\u9fff]|[^\W\u3400-\u9fff]+", re.UNICODE)


@dataclass(frozen=True)
class HistoryRecord:
    id: str
    text: str
    image: Image.Image | None = None
    requires: tuple[str, ...] = ()

    def __post_init__(self):
        if self.text.count(IMAGE) != int(self.image is not None):
            raise ValueError("each image requires exactly one complete placeholder")


@dataclass(frozen=True)
class GUIContext:
    prefix: HistoryRecord
    history: tuple[HistoryRecord, ...]
    suffix: HistoryRecord
    query: str
    required: tuple[str, ...] = ()

    def render(self, history=None):
        records = self.history if history is None else history
        parts = (self.prefix, *records, self.suffix)
        prompt = {"prompt": "".join(record.text for record in parts)}
        images = [record.image for record in parts if record.image is not None]
        if images:
            prompt["multi_modal_data"] = {"image": images[0] if len(images) == 1 else images}
        return prompt


@dataclass(frozen=True)
class Selection:
    history: tuple[HistoryRecord, ...]
    fallback: str | None = None

    @property
    def kept(self):
        return [record.id for record in self.history]


def split_text(text, tokenizer, prefix, limit=256):
    """Split a text-only history run without changing its concatenated bytes."""
    if IMAGE in text:
        raise ValueError("an image and its record must remain atomic")
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    boundaries = [0]
    for start in range(limit, len(offsets), limit):
        boundary = offsets[start][0]
        if boundary > boundaries[-1]:
            boundaries.append(boundary)
    boundaries.append(len(text))
    return tuple(HistoryRecord(f"{prefix}{i}", text[first:last])
                 for i, (first, last) in enumerate(zip(boundaries, boundaries[1:])) if last > first)


def add_dependencies(seed, by_id):
    visited = set()
    pending = list(seed)
    while pending:
        key = pending.pop()
        if key not in visited:
            visited.add(key)
            pending.extend(by_id[key].requires)
    return visited


def bm25(history, query):
    terms = set(WORDS.findall(query.casefold()))
    documents = [Counter(WORDS.findall(record.text.casefold())) for record in history]
    lengths = [sum(doc.values()) for doc in documents]
    mean = sum(lengths) / max(1, len(documents)) or 1
    inverse = {term: log(1 + (len(documents) - df + .5) / (df + .5))
               for term in sorted(terms)
               if (df := sum(term in doc for doc in documents))}
    scores = []
    for doc, length in zip(documents, lengths):
        norm = 1.2 * (.25 + .75 * length / mean)
        scores.append(sum(weight * 2.2 * doc[term] / (doc[term] + norm)
                          for term, weight in inverse.items() if doc[term]))
    return scores


def select_history(context: GUIContext, cost: Callable[[HistoryRecord], int],
                   budget=DEFAULT_BUDGET, strategy="working_set"):
    if budget < 1 or strategy not in STRATEGIES:
        raise ValueError("invalid context budget or strategy")
    history = context.history
    if not history:
        return Selection(history)
    costs = {record.id: cost(record) for record in history}
    fixed = cost(context.prefix) + cost(context.suffix)
    if fixed + sum(costs.values()) <= budget:
        return Selection(history)

    by_id = {record.id: record for record in history}
    if len(by_id) != len(history):
        return Selection(history, "duplicate_record_id")
    protected = {history[-1].id}
    expand = strategy in {"working_set", "oracle"}
    if expand:
        protected.update(context.required)
        references = protected | {key for record in history for key in record.requires}
        if not references <= by_id.keys():
            return Selection(history, "missing_reference")
        protected = add_dependencies(protected, by_id)
    remaining = budget - fixed - sum(costs[key] for key in protected)
    if remaining < 0:
        return Selection(history, "protected_context_exceeds_budget")
    if strategy == "oracle":
        return Selection(tuple(record for record in history if record.id in protected))

    scores = bm25(history, context.query) if strategy != "recent" else [0.] * len(history)
    order = sorted(range(len(history)), key=lambda i: (scores[i], i), reverse=True)
    kept = set(protected)
    for i in order:
        key = history[i].id
        if key in kept:
            continue
        group = (add_dependencies((key,), by_id) if expand else {key}) - kept
        amount = sum(costs[item] for item in group)
        if amount <= remaining:
            kept.update(group)
            remaining -= amount
    return Selection(tuple(record for record in history if record.id in kept))
