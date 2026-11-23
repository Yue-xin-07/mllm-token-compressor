from PIL import Image
import pytest

from compressor.working_set import GUIContext, HistoryRecord, IMAGE, select_history, split_text


def cost(record):
    return 2 if record.image else len(record.text.split())


def context(history, required=()):
    return GUIContext(HistoryRecord("start", "system "),
                      tuple(history), HistoryRecord("end", "query"),
                      "open action blue", required)


def test_protected_dependency_and_original_order():
    records = [HistoryRecord("state", "old state "),
               HistoryRecord("noise", "open blue action distraction "),
               HistoryRecord("call", "resolved action", requires=("state",)),
               HistoryRecord("recent", "latest ")]
    result = select_history(context(records, ("call",)), cost, budget=7)
    assert result.kept == ["state", "call", "recent"]
    assert result.fallback is None
    lexical = select_history(context(records, ("call",)), cost, budget=7, strategy="bm25")
    assert lexical.kept == ["noise", "recent"]


def test_recent_and_bm25_differ_at_equal_budget():
    records = [HistoryRecord("match", "blue action "), HistoryRecord("noise", "random old "),
               HistoryRecord("recent", "latest ")]
    request = context(records)
    assert select_history(request, cost, 5, "recent").kept == ["noise", "recent"]
    assert select_history(request, cost, 5, "bm25").kept == ["match", "recent"]


@pytest.mark.parametrize(("required", "reason"), [
    (("absent",), "missing_reference"),
    (("first",), "protected_context_exceeds_budget"),
])
def test_infeasible_selection_falls_back(required, reason):
    records = [HistoryRecord("first", "history state evidence"), HistoryRecord("last", "latest")]
    result = select_history(context(records, required), cost, 4)
    assert result.fallback == reason
    assert result.kept == ["first", "last"]


def test_image_payload_follows_selected_markers():
    previous = Image.new("RGB", (28, 28), "blue")
    current = Image.new("RGB", (28, 28), "red")
    request = GUIContext(HistoryRecord("start", "system "),
                         (HistoryRecord("old", IMAGE, previous), HistoryRecord("recent", "latest ")),
                         HistoryRecord("current", IMAGE + "which?", current), "which?")
    result = select_history(request, cost, 4)
    assert result.kept == ["recent"]
    assert request.render(result.history)["multi_modal_data"]["image"] is current
    assert request.render(result.history)["prompt"].count(IMAGE) == 1


def test_text_split_retains_source_order_and_bytes():
    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}

    parts = split_text("abc你def", Tokenizer(), "p", 3)
    assert [p.text for p in parts] == ["abc", "你de", "f"]
    assert [p.id for p in parts] == ["p0", "p1", "p2"]



def test_malformed_image_placeholder_is_rejected_before_submission():
    with pytest.raises(ValueError, match="complete placeholder"):
        HistoryRecord("bad", "<|image_pad|>")
    image = Image.new("RGB", (28, 28))
    with pytest.raises(ValueError, match="complete placeholder"):
        HistoryRecord("bad", IMAGE + "<|image_pad|>", image)


def test_short_unsupported_references_report_fallback():
    request = context([HistoryRecord("recent", "latest ")], ("missing",))
    assert select_history(request, cost, 100).fallback == "missing_reference"
