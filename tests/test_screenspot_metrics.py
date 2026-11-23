from scripts.screenspot_metrics import click_result

import pytest


@pytest.mark.parametrize(("text", "point", "correct"), [
    ("(100, 200)", [100., 200.], True),
    ("(300, 400)", [300., 400.], True),
    ("(301, 400)", [301., 400.], False),
    ("(1001, 200)", None, False),
    ("(-1, 200)", None, False),
    ("(100, 200) or (301, 400)", None, False),
    ("no point", None, False),
])
def test_click_result(text, point, correct):
    assert click_result(text, [20, 20, 40, 20], (200, 100)) == (point, correct)
