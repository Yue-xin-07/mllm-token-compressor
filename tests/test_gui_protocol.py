import json
import pytest

from scripts.gui_protocol import score_click, score_completion, score_prediction


def reply(point, *, action="left_click", name="computer_use"):
    call = {"name": name, "arguments": {"action": action, "coordinate": point}}
    return f"<tool_call>{json.dumps(call)}</tool_call>"


@pytest.mark.parametrize(("point", "expected"), [
    ([100, 200], True), ([300, 400], True), ([301, 400], False),
])
def test_gui_click_checks_normalized_pixels_and_box_boundaries(point, expected):
    assert score_click(reply(point), [20, 20, 40, 20], (200, 100)) == (point, expected)


@pytest.mark.parametrize("reply_text", [
    "(100, 200)", reply([-1, 200]), reply([True, 200]), reply([float("nan"), 200]),
    reply([100, 200], action="scroll"), reply([100, 200], name="other"),
    reply([100, 200]) * 2, "<tool_call>{xxx}</tool_call>",
])
def test_rejects_non_click_or_ambiguous_answers(reply_text):
    assert score_click(reply_text, [20, 20, 40, 20], (200, 100)) == (None, False)



def test_prefilled_click_only_scores_a_complete_coordinate_array():
    bbox, size = [20, 20, 40, 20], (200, 100)
    assert score_completion("200, 300]", bbox, size) == ([200, 300], True)
    assert score_completion("200, 300", bbox, size) == (None, False)
    assert score_completion("200, 300, 400]", bbox, size) == (None, False)



@pytest.mark.parametrize(("completion", "protocol"), [("140, 56]", "point"), ("126,42,154,70]", "bbox")])
def test_resized_pixel_protocols_converge_on_the_same_target(completion, protocol):
    point, correct = score_prediction(completion, [126, 42, 28, 28], (280, 280), protocol)
    assert point == [500, 200]
    assert correct


def test_rejects_out_of_range_resized_pixel_prediction():
    assert score_prediction("281, 56]", [0, 0, 280, 280], (280, 280), "point") == (None, False)
