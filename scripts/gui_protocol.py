"""Offline pointing protocol compatible with Qwen's computer-use tool format."""
from copy import deepcopy
import json
import math
import re

from PIL import Image

from compressor.working_set import GUIContext, HistoryRecord, IMAGE


IMAGE_PIXELS = 1024 * 28 * 28
PROTOCOLS = ("tool", "point", "bbox")
BBOX_PREFIX = '[{"bbox_2d": ['
BBOX_SYSTEM = "<|im_start|>system\nYou are a visual grounding assistant.<|im_end|>\n"
BBOX_COMPLETION = re.compile(r"\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\s*")


TOOL = {
    "type": "function",
    "function": {
        "name": "computer_use",
        "description": "Interact with a 1000 by 1000 screen. Click the center of the requested element.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["left_click"]},
                "coordinate": {
                    "type": "array",
                    "description": "Horizontal and vertical position from the top-left, scaled to 0–1000.",
                    "items": {"type": "number"}, "minItems": 2, "maxItems": 2,
                },
            },
            "required": ["action", "coordinate"],
        },
    },
}
SYSTEM = (
    "<|im_start|>system\nYou are a GUI assistant. Ground the requested action in the latest screenshot.\n"
    "<tools>\n" + json.dumps(TOOL, ensure_ascii=False) + "\n</tools>\n"
    "Reply with one JSON object containing the function name and arguments, enclosed in "
    "<tool_call> and </tool_call> tags.<|im_end|>\n"
)
CLICK_PREFIX = '<tool_call>\n{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": ['
COMPLETION = re.compile(r"\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]\s*")
CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def pixel_size(size):
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    height, width = smart_resize(size[1], size[0], factor=28,
                                 min_pixels=3136, max_pixels=IMAGE_PIXELS)
    return width, height


def pixel_tool_system(size):
    width, height = pixel_size(size)
    tool = deepcopy(TOOL)
    schema = tool["function"]
    schema["description"] = f"Click a desktop GUI whose resolution is {width} by {height} pixels."
    schema["parameters"]["properties"]["coordinate"]["description"] = (
        "Horizontal and vertical pixel coordinates measured from the top-left of the resized screen.")
    return SYSTEM.replace(json.dumps(TOOL, ensure_ascii=False), json.dumps(tool, ensure_ascii=False))


def make_click_context(sample, *, history=(), query=None, required=(), protocol="tool"):
    with Image.open(sample["image"]) as source:
        image = source.convert("RGB")
    query = query or sample["instruction"]
    if protocol == "bbox":
        instruction = ("Find the UI element needed for the current instruction. "
                       "Output its bounding box [left, top, right, bottom] in resized-image pixels. "
                       f"Instruction: {query}")
        system, prefill = BBOX_SYSTEM, BBOX_PREFIX
    elif protocol == "point":
        instruction = f"Click the center of the GUI element for this instruction: {query}"
        system, prefill = pixel_tool_system(image.size), CLICK_PREFIX
    elif protocol == "tool":
        instruction = f"Use one left click to perform this instruction: {query}"
        system, prefill = SYSTEM, CLICK_PREFIX
    else:
        raise ValueError(f"unknown GUI protocol {protocol}")
    suffix = ("<|im_start|>user\n" + IMAGE + "\n" + instruction +
              "<|im_end|>\n<|im_start|>assistant\n" + prefill)
    return GUIContext(HistoryRecord("system", system), tuple(history),
                      HistoryRecord("current-screen", suffix, image), query, required)


def score_click(text, bbox, size):
    matches = CALL.findall(text)
    if len(matches) != 1:
        return None, False
    try:
        tool = json.loads(matches[0])
        args = tool["arguments"]
        if tool["name"] != "computer_use" or args["action"] != "left_click":
            return None, False
        point = args["coordinate"]
        if not isinstance(point, list) or len(point) != 2:
            return None, False
        if not all(type(value) in (int, float) and 0 <= value <= 1000
                   and math.isfinite(value) for value in point):
            return None, False
    except (KeyError, TypeError, ValueError):
        return None, False
    left, top, width, height = bbox
    x, y = [value * extent / 1000 for value, extent in zip(point, size, strict=True)]
    return point, left <= x <= left + width and top <= y <= top + height



def score_completion(text, bbox, size):
    """Only coordinates are sampled. The shared assistant prefix fixes the tool schema."""
    match = COMPLETION.fullmatch(text)
    if match is None:
        return None, False
    return score_click(CLICK_PREFIX + text + "}}</tool_call>", bbox, size)



def score_prediction(text, bbox, size, protocol="tool"):
    if protocol == "tool":
        return score_completion(text, bbox, size)
    width, height = pixel_size(size)
    if protocol == "point":
        match = COMPLETION.fullmatch(text)
        if match is None:
            return None, False
        x, y = map(float, match.groups())
        if not (0 <= x <= width and 0 <= y <= height):
            return None, False
        point = [x * 1000 / width, y * 1000 / height]
    elif protocol == "bbox":
        match = BBOX_COMPLETION.fullmatch(text)
        if match is None:
            return None, False
        left, top, right, bottom = map(int, match.groups())
        if not (0 <= left <= right <= width and 0 <= top <= bottom <= height):
            return None, False
        point = [(left + right) * 500 / width, (top + bottom) * 500 / height]
    else:
        raise ValueError(f"unknown GUI protocol {protocol}")
    call = {"name": "computer_use", "arguments": {"action": "left_click", "coordinate": point}}
    return score_click("<tool_call>" + json.dumps(call) + "</tool_call>", bbox, size)
