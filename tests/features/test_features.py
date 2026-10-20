import numpy as np
import pytest

from wakesmith.config import CHUNK_SAMPLES
from wakesmith.features import held_out, place, to_int16, windows


def test_windows_never_cross_recordings():
    frames = np.arange(50, dtype=np.float32)[:, None].repeat(96, 1)
    out = windows(frames, np.array([0, 20, 30, 50]), stride=1)
    assert len(out) == 10
    assert all(w[-1, 0] - w[0, 0] == 15 for w in out)


def test_place_end_aligns_with_trailing_context():
    audio = np.ones(1000, np.float32)
    out = place(audio, 32000, np.random.default_rng(0), jitter=0)
    assert out[-1] == 1 and out[-1001] == 0


def test_held_out_split_is_stable_and_roughly_ten_percent():
    names = [f"noise-{i}.wav" for i in range(2000)]
    share = sum(map(held_out, names)) / len(names)
    assert 0.05 < share < 0.15
    assert held_out(names[0]) == held_out(names[0])


def test_streaming_features_equal_the_runtime_preprocessor():
    pytest.importorskip("openwakeword")
    from openwakeword.utils import AudioFeatures

    from wakesmith.features import stream_embeddings

    try:
        runtime = AudioFeatures(inference_framework="onnx")
    except Exception:
        pytest.skip("openWakeWord feature models are not downloaded")
    t = np.arange(32000) / 16000
    pcm = to_int16(0.3 * np.sin(2 * np.pi * (200 + 300 * t) * t))
    stream_embeddings(
        to_int16(np.random.default_rng(0).uniform(-0.5, 0.5, 20000))
    )  # No state leaks.
    ours = stream_embeddings(pcm)[-16:]
    runtime.reset()
    for _ in range(40):
        runtime(np.zeros(CHUNK_SAMPLES, np.int16))
    for i in range(0, len(pcm), CHUNK_SAMPLES):
        runtime(pcm[i : i + CHUNK_SAMPLES])
    np.testing.assert_allclose(runtime.get_features(16)[0], ours, rtol=1e-5, atol=1e-4)
