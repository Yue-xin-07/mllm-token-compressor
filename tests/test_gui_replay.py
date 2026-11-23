from PIL import Image

from compressor.working_set import IMAGE
from scripts.gui_replay import build_replay


def test_replay_needs_explicit_older_evidence_and_protects_current_screen(tmp_path):
    current = tmp_path / "current.png"
    old = tmp_path / "old.png"
    Image.new("RGB", (28, 28), "red").save(current)
    Image.new("RGB", (28, 28), "blue").save(old)
    sample = {"id": "desktop/3", "image": current, "instruction": "open the red tab"}
    context, (ticket, evidence) = build_replay(sample, old)
    index = {record.id: record for record in context.history}
    assert ticket in context.query
    assert context.required == (ticket,)
    assert index[ticket].requires == (evidence,)
    assert sample["instruction"] in index[evidence].text
    assert sample["instruction"] not in context.query
    assert context.render()["prompt"].count(IMAGE) == 2
    assert context.suffix.image.getpixel((0, 0)) == (255, 0, 0)
