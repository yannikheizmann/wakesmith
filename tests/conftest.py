from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def feature_dir(tmp_path: Path) -> tuple[Path, Path]:
    """Tiny, separable feature files in the layout the training stage expects."""
    rng = np.random.default_rng(0)
    features = tmp_path / "features"
    features.mkdir()

    def windows(n: int, shift: float) -> np.ndarray:
        return (rng.standard_normal((n, 16, 96)) + shift).astype(np.float16)

    for split, n in (("train", 256), ("validation", 64)):
        np.save(features / f"kokoro_positive_{split}.npy", windows(n, 2.0))
        np.save(features / f"kokoro_adversarial_{split}.npy", windows(n, -1.0))
    np.save(features / "background_train.npy", windows(128, -2.0))
    np.save(
        features / "owwval_validation.npy", (rng.standard_normal((2000, 96)) - 1).astype(np.float16)
    )
    frames = (rng.standard_normal((900, 96)) - 1).astype(np.float16)
    np.savez(features / "speech_train.npz", frames=frames, offsets=np.array([0, 400, 900]))
    np.savez(features / "speech_validation.npz", frames=frames[:500], offsets=np.array([0, 500]))
    acav = tmp_path / "acav.npy"
    np.save(acav, windows(4096, -1.0))
    return features, acav
