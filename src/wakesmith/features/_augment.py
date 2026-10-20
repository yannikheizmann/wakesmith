"""Augment clips with equalisation, distortion, pitch, noise, rooms and level changes."""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np

from wakesmith.config import SAMPLE_RATE


def held_out(path: str | Path, fraction: float = 0.1) -> bool:
    """Assign a stable tenth of files to evaluation, so test noise and rooms stay unseen."""
    return zlib.crc32(Path(path).name.encode()) % 1000 < fraction * 1000


def noise_sources(musan: Path, rirs: Path, evaluation: bool = False) -> tuple[list[str], list[str]]:
    """Background recordings and room impulse responses for one side of the split."""
    backgrounds = sorted(str(p) for p in Path(musan).glob("**/*.wav") if held_out(p) == evaluation)
    impulses = sorted(str(p) for p in Path(rirs).glob("*.wav") if held_out(p) == evaluation)
    return backgrounds, impulses


def read_mono(path: str | Path) -> np.ndarray:
    """Read the first channel of an audio file at 16 kHz as float32."""
    import soundfile as sf

    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio[:, 0]
    if rate != SAMPLE_RATE:
        from scipy.signal import resample_poly

        audio = resample_poly(audio, SAMPLE_RATE, rate).astype(np.float32)
    return audio


def place(
    audio: np.ndarray, length: int, rng: np.random.Generator, jitter: int = 3200
) -> np.ndarray:
    """End-align a clip in a fixed window, leaving 0-200 ms of trailing context."""
    out = np.zeros(length, dtype=np.float32)
    end = length - int(rng.integers(0, jitter + 1))
    audio = audio[-end:]
    out[end - len(audio) : end] = audio
    return out


def to_int16(audio: np.ndarray) -> np.ndarray:
    return (np.clip(audio, -1, 1) * 32767).astype(np.int16)


class Augmentor:
    """openWakeWord's augmentation chain with wider level and signal-to-noise coverage."""

    def __init__(
        self,
        backgrounds: list[str],
        impulses: list[str],
        rng: np.random.Generator,
        snr: tuple[float, float] = (-5.0, 20.0),
    ):
        import audiomentations as A

        self.backgrounds, self.impulses, self.rng, self.snr = backgrounds, impulses, rng, snr
        self.chain = A.Compose(
            [
                A.SevenBandParametricEQ(min_gain_db=-6, max_gain_db=6, p=0.25),
                A.TanhDistortion(min_distortion=0.0001, max_distortion=0.10, p=0.25),
                A.PitchShift(min_semitones=-3, max_semitones=3, p=0.25),
                A.BandStopFilter(p=0.25),
                A.AddColorNoise(min_snr_db=10, max_snr_db=30, p=0.25),
            ]
        )
        self._lengths: dict[str, tuple[int, int]] = {}

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        audio = self.chain(samples=audio, sample_rate=SAMPLE_RATE)
        return self.level(self.add_background(self.reverberate(audio)))

    def background(self, length: int) -> np.ndarray:
        """A random excerpt, read without loading whole minutes-long recordings."""
        import soundfile as sf

        path = self.backgrounds[self.rng.integers(len(self.backgrounds))]
        if path not in self._lengths:
            info = sf.info(path)
            self._lengths[path] = (info.frames, info.samplerate)
        frames, rate = self._lengths[path]
        need = int(np.ceil(length * rate / SAMPLE_RATE))
        start = int(self.rng.integers(0, max(1, frames - need + 1)))
        audio = sf.read(path, start=start, frames=need, dtype="float32", always_2d=True)[0][:, 0]
        if rate != SAMPLE_RATE:
            from scipy.signal import resample_poly

            audio = resample_poly(audio, SAMPLE_RATE, rate).astype(np.float32)
        if len(audio) < length:
            audio = np.tile(audio, length // max(1, len(audio)) + 1)
        return audio[:length]

    def reverberate(self, audio: np.ndarray, p: float = 0.5) -> np.ndarray:
        if not self.impulses or self.rng.random() > p:
            return audio
        from scipy.signal import fftconvolve

        rir = read_mono(self.impulses[self.rng.integers(len(self.impulses))])
        wet = fftconvolve(audio, rir / (np.abs(rir).max() + 1e-8))[: len(audio)]
        return (wet * (np.abs(audio).max() / (np.abs(wet).max() + 1e-8))).astype(np.float32)

    def add_background(self, audio: np.ndarray, p: float = 0.75) -> np.ndarray:
        if not self.backgrounds or self.rng.random() > p:
            return audio
        noise = self.background(len(audio))
        snr = self.rng.uniform(*self.snr)
        scale = np.sqrt(
            (np.mean(audio**2) + 1e-10) / ((np.mean(noise**2) + 1e-10) * 10 ** (snr / 10))
        )
        return (audio + scale * noise).astype(np.float32)

    def level(self, audio: np.ndarray) -> np.ndarray:
        """Random peak level between -36 and -1 dBFS: quiet far field to close talk."""
        return audio * (10 ** (self.rng.uniform(-36, -1) / 20) / (np.abs(audio).max() + 1e-8))
