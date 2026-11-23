"""Load and score the original SeeClick ScreenSpot test annotations."""
import hashlib
import json
import math
import re
from pathlib import Path


SPLITS = ("desktop", "mobile", "web")
POINT = re.compile(r"\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)")


def load_dataset(root):
    samples = []
    hashes = {}
    images = root / "screenspot_imgs"
    if not images.is_dir():
        images = root
    for split in SPLITS:
        path = root / f"screenspot_{split}.json"
        raw = path.read_bytes()
        hashes[path.name] = hashlib.sha256(raw).hexdigest()
        for index, row in enumerate(json.loads(raw)):
            filename = row["img_filename"]
            if Path(filename).name != filename:
                raise ValueError(f"invalid ScreenSpot image name: {filename}")
            image = images / filename
            if not image.is_file():
                raise FileNotFoundError(image)
            samples.append({
                "id": f"{split}/{index}",
                "split": split,
                "image": image,
                "instruction": row["instruction"],
                "bbox": row["bbox"],
                "type": row["data_type"],
            })
    return samples, hashes


def click_result(text, bbox, size):
    """Score one normalized point against the original pixel [x, y, w, h] box."""
    matches = POINT.findall(text)
    if len(matches) != 1:
        return None, False
    x, y = map(float, matches[0])
    if not all(math.isfinite(v) and 0 <= v <= 1000 for v in (x, y)):
        return None, False
    left, top, width, height = bbox
    px, py = x * size[0] / 1000, y * size[1] / 1000
    return [x, y], left <= px <= left + width and top <= py <= top + height
