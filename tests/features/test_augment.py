import numpy as np
import soundfile as sf

from wakesmith.features import Augmentor, noise_sources, read_mono


def test_augmentor_keeps_length_and_bounded_level(tmp_path):
    musan, rirs = tmp_path / "musan" / "noise", tmp_path / "rirs"
    musan.mkdir(parents=True)
    rirs.mkdir()
    rng = np.random.default_rng(0)
    for i in range(30):
        sf.write(musan / f"n{i}.wav", rng.standard_normal(48000).astype(np.float32) * 0.1, 48000)
        impulse = np.exp(-np.arange(1600) / 200) * rng.standard_normal(1600)
        sf.write(rirs / f"r{i}.wav", impulse.astype(np.float32), 16000)
    sources = noise_sources(tmp_path / "musan", rirs)
    held = noise_sources(tmp_path / "musan", rirs, evaluation=True)
    assert not set(sources[0]) & set(held[0])
    augment = Augmentor(*sources, rng)
    speech = (0.3 * np.sin(np.arange(32000) / 7)).astype(np.float32)
    for _ in range(5):
        out = augment(speech.copy())
        assert out.shape == speech.shape and np.isfinite(out).all()
        assert np.abs(out).max() <= 0.9
    assert augment.background(5000).shape == (5000,)
    assert read_mono(sources[0][0]).shape == (16000,)
