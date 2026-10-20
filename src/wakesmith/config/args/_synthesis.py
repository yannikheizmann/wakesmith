"""Define text-to-speech and ASR screening arguments."""

from __future__ import annotations

from pydantic import Field, NonNegativeInt, PositiveInt

from .._constants import (
    KOKORO_EVALUATION_VOICES,
    KOKORO_VOICES,
    VOXCPM_EVALUATION_VOICES,
    VOXCPM_TRAIN_VOICES,
)
from ._base import ArgsBase


class KokoroArgs(ArgsBase):
    """Kokoro-82M synthesis: blended voice embeddings, the main source of positives."""

    n_train: NonNegativeInt = Field(default=24000, description="Positive training clips.")
    n_validation: NonNegativeInt = Field(default=1500, description="Positive validation clips.")
    n_evaluation: NonNegativeInt = Field(default=800, description="Held-out-voice positives.")
    n_adversarial_train: NonNegativeInt = Field(
        default=12000, description="Confusable training clips."
    )
    n_adversarial_validation: NonNegativeInt = Field(
        default=1000, description="Confusable validation clips."
    )
    accents: dict[str, float] = Field(
        default={"a": 0.45, "b": 0.25, "e": 0.06, "f": 0.06, "i": 0.06, "p": 0.06, "h": 0.06},
        description="English only: front ends (US, UK, es, fr, it, pt, hi) and their weights.",
    )
    speed: tuple[float, float] = Field(default=(0.75, 1.25), description="Speaking-rate range.")
    voices: tuple[str, ...] = Field(
        default=KOKORO_VOICES, description="Voices available for blends."
    )
    evaluation_voices: tuple[str, ...] = Field(
        default=KOKORO_EVALUATION_VOICES, description="Voices excluded from training blends."
    )


class VoxCPMArgs(ArgsBase):
    """VoxCPM2 voice design: free-text speaker descriptions in 30 languages."""

    enabled: bool = Field(default=True, description="Synthesise with VoxCPM2 (needs ~12 GB RAM).")
    n_train: NonNegativeInt = 5000
    n_validation: NonNegativeInt = 300
    n_evaluation: NonNegativeInt = 500
    n_adversarial_train: NonNegativeInt = 2000
    n_adversarial_validation: NonNegativeInt = 200
    workers: PositiveInt = Field(default=1, description="Parallel VoxCPM processes.")
    cfg_values: tuple[float, ...] = (1.6, 2.0, 2.6)
    inference_timesteps: tuple[int, ...] = (8, 10)
    train_voices: tuple[str, ...] = VOXCPM_TRAIN_VOICES
    evaluation_voices: tuple[str, ...] = VOXCPM_EVALUATION_VOICES


class PiperArgs(ArgsBase):
    """Piper LibriTTS (English only): 904 speakers for confusable phrases."""

    n_adversarial_train: NonNegativeInt = 30000
    n_adversarial_validation: NonNegativeInt = 2000
    batch_size: PositiveInt = 128
    noise_scales: tuple[float, ...] = (0.667, 0.85, 0.98)
    noise_scale_ws: tuple[float, ...] = (0.8, 0.98)
    length_scales: tuple[float, ...] = (0.75, 0.9, 1.0, 1.15, 1.3)
    slerp_weights: tuple[float, ...] = (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)


class ScreenArgs(ArgsBase):
    """Whisper screening of synthetic clips."""

    model: str = "openai/whisper-large-v3-turbo"
    revision: str = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
    batch_size: PositiveInt = 64
    vowel_tolerance: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Maximum normalised vowel edit distance for a transcript to match.",
    )
