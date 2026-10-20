import os

import pytest

from wakesmith.pipeline import Display


def test_stage_output_goes_to_its_log(tmp_path):
    with Display("title", "subtitle", {"one": "One", "two": "Two"}, tmp_path) as display:
        with display.stage("one"):
            os.write(1, b"from a library\n")
            display.progress(1, 2, "items")
        with display.stage("two", cached=True):
            pass
        assert display.stages["one"].state == "done" and display.stages["two"].state == "cached"
    assert "from a library" in (tmp_path / "one.log").read_text()


def test_failed_stage_is_marked(tmp_path):
    with Display("title", "subtitle", {"one": "One"}, tmp_path) as display:
        with pytest.raises(RuntimeError), display.stage("one"):
            raise RuntimeError("boom")
        assert display.stages["one"].state == "failed"
