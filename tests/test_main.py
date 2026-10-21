import re

from typer.testing import CliRunner

from wakesmith.main import app

runner = CliRunner()

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def test_help_lists_the_options():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    # Renderers differ across environments (styles, panel layouts, word folding);
    # match on the text itself, independent of styling and whitespace.
    compact = re.sub(r"\s+", "", ANSI.sub("", result.output))
    assert "--language" in compact and "--config" in compact


def test_invalid_wake_word_is_a_usage_error(tmp_path):
    result = runner.invoke(app, ["1", "--work-dir", str(tmp_path)])
    assert result.exit_code == 2
