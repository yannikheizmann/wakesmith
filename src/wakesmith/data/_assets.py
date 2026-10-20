"""Locate pinned external inputs inside the shared cache directory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# (repository, revision, kind); revisions are pinned so every run sees identical data.
FEATURES = (
    "davidscripka/openwakeword_features",
    "985bf1b47e7f19c07741af82bfe32d5a9dc56096",
    "dataset",
)
MUSAN = ("FluidInference/musan", "3edcfdf89b56dbe6a395ff29f9c29489e03d1321", "dataset")
RIRS = (
    "davidscripka/MIT_environmental_impulse_responses",
    "b824a1ef2821f112fda0b9cb26e4278c62b425bb",
    "dataset",
)
MLS = ("facebook/multilingual_librispeech", "2e83e61823b4c47dcbcb1980bb88601274127609", "dataset")
VOXCPM = ("openbmb/VoxCPM2", "32279effe8c19989596f05d353d1447f51d9e915", "model")
KOKORO = ("hexgrad/Kokoro-82M", "f3ff3571791e39611d31c381e3a41a3af07b4987", "model")
PIPER = "https://github.com/livekit/livekit-wakeword/releases/download/v0.1.0/en-us-libritts-high"
OPENWAKEWORD = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"


@dataclass(frozen=True)
class Assets:
    """Paths of every dataset and model weight below one cache root."""

    root: Path

    @property
    def acav(self) -> Path:
        """2,000 h of precomputed English speech features (negatives)."""
        return self.root / "features" / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"

    @property
    def validation(self) -> Path:
        """~11 h of openWakeWord's validation speech features."""
        return self.root / "features" / "validation_set_features.npy"

    @property
    def musan(self) -> Path:
        return self.root / "musan"

    @property
    def rirs(self) -> Path:
        return self.root / "rirs" / "16khz"

    @property
    def mls(self) -> Path:
        return self.root / "mls"

    @property
    def kokoro(self) -> Path:
        return self.root / "kokoro"

    @property
    def voxcpm(self) -> Path:
        return self.root / "voxcpm2"

    @property
    def piper(self) -> Path:
        return self.root / "piper" / "en-us-libritts-high.pt"

    @property
    def openwakeword(self) -> Path:
        import openwakeword

        return Path(openwakeword.__file__).parent / "resources" / "models"

    def mls_shards(self, subset: str, train_shards: int) -> dict[str, list[Path]]:
        """Speaker-disjoint MLS splits: train shards, dev (validation), test (evaluation)."""
        base = self.mls / subset
        return {
            "train": [next(base.glob(f"train-{i:05d}-of-*.parquet")) for i in range(train_shards)],
            "validation": sorted(base.glob("dev-*.parquet")),
            "evaluation": sorted(base.glob("test-*.parquet")),
        }
