"""Define negative-speech and augmentation arguments."""

from __future__ import annotations

from pydantic import Field, NonNegativeInt, PositiveFloat, PositiveInt, field_validator

from .._constants import LANGUAGES
from ._base import ArgsBase


class NegativeSpeechArgs(ArgsBase):
    """Multilingual LibriSpeech speech added to the 2,000 h of English ACAV100M negatives."""

    language: str | None = Field(
        default=None,
        description="Language code; defaults to the wake-word language when it is not English.",
    )
    train_shards: PositiveInt = Field(default=2, description="MLS training shards (~30 h each).")
    train_hours: PositiveFloat = 60
    validation_hours: PositiveFloat = 6
    evaluation_hours: PositiveFloat = 6

    @field_validator("language")
    @classmethod
    def _has_corpus(cls, value: str | None) -> str | None:
        if value is not None and (value not in LANGUAGES or LANGUAGES[value].mls is None):
            usable = sorted(k for k, v in LANGUAGES.items() if v.mls)
            raise ValueError(f"no negative-speech corpus for {value!r}; choose from {usable}")
        return value


class AugmentationArgs(ArgsBase):
    """Augmented copies per clip and standalone noise negatives."""

    rounds: PositiveInt = Field(default=3, description="Augmented copies of each training clip.")
    validation_rounds: PositiveInt = 1
    background_clips: NonNegativeInt = Field(default=4000, description="Pure noise/music windows.")
