"""Deterministic GUI replays with labeled dependencies and real screenshots."""
from hashlib import sha256
from PIL import Image

from compressor.working_set import HistoryRecord, IMAGE
from scripts.gui_protocol import make_click_context


NOTES = (
    "The upload modal opened. The staging office reviewed the activity log and left the pending action unchanged.",
    "The notifications list refreshed. No approval was recorded. The browser kept the workspace open.",
    "The settings panel was inspected. The active status remained pending and was filed in the work queue.",
    "The sidebar opened and closed. The workspace reviewer recorded the transient loading state.",
    "The account panel returned a permission hint. The active workflow stayed on hold until confirmation.",
    "The browser fetched a stale status from the service. The reviewer marked it as a retry and continued.",
)


def message(text):
    return "<|im_start|>user\n" + text + "<|im_end|>\n"


def build_replay(sample, older_screenshot, protocol="tool"):
    ticket = "T-" + sha256(sample["id"].encode()).hexdigest()[:8]
    source = "result-" + ticket
    directive = sample["instruction"].rstrip(".")
    evidence = HistoryRecord(
        source, message(f"Approved tool result {source}. On the new screenshot, the next required "
                        f"click instruction is: {directive}."))
    entry = HistoryRecord(ticket, message(f"Ticket {ticket} links to approved tool result {source}. "
                                          "Read that result to learn the target."),
                          requires=(source,))
    with Image.open(older_screenshot) as previous:
        image = previous.convert("RGB")
    frame = HistoryRecord("previous-screen", message("Archived screen from another task. " + IMAGE),
                          image)
    log = []
    for i in range(92):
        text = f"Audit entry {i:03d}, other workflow. {NOTES[i % len(NOTES)]}"
        log.append(HistoryRecord(f"log-{i}", message(text)))
    history = tuple([frame, *log[:9], evidence, *log[9:17], entry, *log[17:]])
    query = (f"Resume ticket {ticket}. Read its approved tool result to identify the element, "
             "then click its center in the current screenshot.")
    return make_click_context(sample, history=history, query=query, required=(ticket,), protocol=protocol), (ticket, source)
