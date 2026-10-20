"""Write, count and resume synthetic clips.

Clip ``i`` of a split is a pure function of ``(seed, split, i)`` and files are published
atomically, so an interrupted run resumes exactly where it stopped.
"""

from __future__ import annotations

import re
import zlib
from pathlib import Path

import numpy as np

from wakesmith.config import SAMPLE_RATE

_CLIP = re.compile(r"^clip_\d{6}\.wav$")


def clips(directory: Path) -> list[Path]:
    """Completed clips of a split, in index order."""
    return sorted(p for p in Path(directory).glob("clip_*.wav") if _CLIP.match(p.name))


def count(directory: Path) -> int:
    return len(clips(directory)) if Path(directory).exists() else 0


def missing(directory: Path, n: int, worker: int = 0, workers: int = 1) -> list[int]:
    """Indices still to synthesise, optionally striped across workers."""
    return [
        i for i in range(worker, n, workers) if not (Path(directory) / f"clip_{i:06d}.wav").exists()
    ]


def split_seed(seed: int, name: str) -> int:
    return seed + zlib.crc32(name.encode()) % 100_000


def trim_edges(audio: np.ndarray, top_db: float = 40) -> np.ndarray:
    """Trim leading and trailing silence only, keeping 50 ms around the speech."""
    import librosa

    audio = np.asarray(audio)
    scale = 32768.0 if audio.dtype == np.int16 else 1.0
    _, (start, end) = librosa.effects.trim(audio.astype(np.float32) / scale, top_db=top_db)
    return audio[max(0, start - 800) : end + 800]


def write_clip(path: Path, audio: np.ndarray, rate: int) -> bool:
    """Resample, trim and peak-normalise; reject empty, overlong or non-finite output."""
    import librosa
    import soundfile as sf

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if rate != SAMPLE_RATE and len(audio):
        audio = librosa.resample(audio, orig_sr=rate, target_sr=SAMPLE_RATE)
    if len(audio):
        audio = trim_edges(audio)
    if (
        len(audio) < SAMPLE_RATE // 5
        or len(audio) > 4 * SAMPLE_RATE
        or not np.isfinite(audio).all()
    ):
        return False
    partial = path.with_suffix(".partial")
    sf.write(
        partial,
        (audio / (np.abs(audio).max() + 1e-8) * 30000).astype(np.int16),
        SAMPLE_RATE,
        format="WAV",
    )
    partial.replace(path)
    return True
