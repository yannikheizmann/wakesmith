from unittest.mock import MagicMock, patch

import pytest

from wakesmith.data import Assets, download


def _response(status: int, chunks: list[bytes]):
    response = MagicMock()
    response.status_code = status
    response.iter_content.return_value = chunks
    response.raise_for_status.side_effect = None if status < 400 else RuntimeError(status)
    response.__enter__.return_value = response
    return response


def test_download_resumes_and_publishes_atomically(tmp_path):
    target = tmp_path / "file.bin"
    (tmp_path / "file.bin.partial").write_bytes(b"abc")
    session = MagicMock()
    session.get.return_value = _response(206, [b"def"])
    with patch("wakesmith.data._download._session", return_value=session):
        download("https://example.invalid/file", target)
    assert target.read_bytes() == b"abcdef"
    assert session.get.call_args.kwargs["headers"] == {"Range": "bytes=3-"}


def test_failed_download_never_publishes(tmp_path):
    session = MagicMock()
    session.get.return_value = _response(500, [])
    with (
        patch("wakesmith.data._download._session", return_value=session),
        pytest.raises(RuntimeError),
    ):
        download("https://example.invalid/file", tmp_path / "file.bin")
    assert not (tmp_path / "file.bin").exists()


def test_asset_layout(tmp_path):
    assets = Assets(tmp_path)
    assert assets.acav.parent == assets.validation.parent == tmp_path / "features"
    assert assets.rirs == tmp_path / "rirs" / "16khz"
