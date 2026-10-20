from typer.testing import CliRunner

from wakesmith.main import app

runner = CliRunner()


def test_help_lists_the_options():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--language" in result.output and "--config" in result.output


def test_invalid_wake_word_is_a_usage_error(tmp_path):
    result = runner.invoke(app, ["1", "--work-dir", str(tmp_path)])
    assert result.exit_code == 2
