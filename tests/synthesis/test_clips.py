import numpy as np
import soundfile as sf

from wakesmith.synthesis import clips, count, write_clip
from wakesmith.synthesis._clips import missing, split_seed


def test_write_clip_trims_normalises_and_rejects_bad_audio(tmp_path):
    audio = np.concatenate([np.zeros(8000), 0.1 * np.sin(np.arange(8000) / 5), np.zeros(8000)])
    assert write_clip(tmp_path / "clip_000000.wav", audio, 16000)
    written, rate = sf.read(tmp_path / "clip_000000.wav")
    assert rate == 16000 and len(written) < len(audio) and np.abs(written).max() > 0.85
    assert not write_clip(tmp_path / "clip_000001.wav", np.zeros(100), 16000)
    assert not write_clip(tmp_path / "clip_000002.wav", np.full(16000, np.nan), 16000)


def test_resume_bookkeeping(tmp_path):
    for i in (0, 2):
        (tmp_path / f"clip_{i:06d}.wav").write_bytes(b"")
    (tmp_path / "clip_000003.partial").write_bytes(b"")
    assert count(tmp_path) == 2 and [p.name for p in clips(tmp_path)] == [
        "clip_000000.wav",
        "clip_000002.wav",
    ]
    assert missing(tmp_path, 4) == [1, 3]
    assert missing(tmp_path, 4, worker=1, workers=2) == [1, 3]
    assert split_seed(1, "a") == split_seed(1, "a") != split_seed(1, "b")
