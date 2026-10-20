"""Compute openWakeWord embeddings exactly as a streaming runtime does.

openWakeWord's training args extracts features in batch, which differs from its own
streaming inference at chunk edges. livekit-wakeword feeds ``[-1, 1]`` floats into the mel
model instead of int16-scale samples, which yields embeddings on another scale entirely.
Streaming through ``AudioFeatures`` avoids both mismatches.
"""

from __future__ import annotations

import numpy as np

from wakesmith.config import CHUNK_SAMPLES

_FEATURES = None
_PRIMED: dict | None = None
_STATE = ("melspectrogram_buffer", "feature_buffer", "accumulated_samples", "raw_data_remainder")


def _primed_features():
    """Return this process's ``AudioFeatures`` after runtime-style silence priming."""
    global _FEATURES, _PRIMED
    if _FEATURES is None:
        from openwakeword.utils import AudioFeatures

        _FEATURES = AudioFeatures(inference_framework="onnx", ncpu=1)
        _FEATURES.reset()
        for _ in range(40):  # 3.2 s of silence after a reset, as in streaming integrations.
            _FEATURES(np.zeros(CHUNK_SAMPLES, np.int16))
        _PRIMED = {key: getattr(_FEATURES, key) for key in _STATE}
        _PRIMED["raw_data_buffer"] = list(_FEATURES.raw_data_buffer)
    return _FEATURES


def stream_embeddings(pcm: np.ndarray, phase: int = 0) -> np.ndarray:
    """Return every embedding frame (one per 80 ms) a runtime produces for int16 audio.

    ``phase`` leading silent samples move the chunk boundaries relative to the speech, as
    happens with a live microphone.
    """
    features = _primed_features()
    for key, value in _PRIMED.items():
        if key == "raw_data_buffer":
            features.raw_data_buffer.clear()
            features.raw_data_buffer.extend(value)
        else:
            setattr(features, key, value.copy() if isinstance(value, np.ndarray) else value)
    pcm = np.concatenate([np.zeros(phase, np.int16), pcm])
    pcm = np.pad(pcm, (0, -len(pcm) % CHUNK_SAMPLES))
    frames = []
    for start in range(0, len(pcm), CHUNK_SAMPLES):
        features(pcm[start : start + CHUNK_SAMPLES])
        frames.append(features.feature_buffer[-1].copy())
    return np.stack(frames)
