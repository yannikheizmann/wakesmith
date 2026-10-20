"""Compose all argument groups into the root configuration of a training run."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator, model_validator

from .._constants import LANGUAGES, Language
from ._base import ArgsBase
from ._data import AugmentationArgs, NegativeSpeechArgs
from ._evaluation import EvaluationArgs
from ._synthesis import KokoroArgs, PiperArgs, ScreenArgs, VoxCPMArgs
from ._training import TrainingArgs


class Args(ArgsBase):
    """Composition root of every setting that determines a run.

    The defaults are the full recipe for a deployable model. Values resolve in order: field
    defaults, an optional YAML file and explicit overrides. A hash of the resolved
    arguments names the run directory, so changing any value starts a new run.
    """

    wake_word: str = Field(description="Word or short phrase to detect.")
    language: str = Field(default="en", description="Language the wake word is spoken in.")
    seed: int = Field(
        default=20261019, description="Seed for all synthesis and training randomness."
    )
    extra_negative_phrases: tuple[str, ...] = Field(
        default=(), description="Additional phrases that must not activate the model."
    )
    kokoro: KokoroArgs = Field(default_factory=KokoroArgs)
    voxcpm: VoxCPMArgs = Field(default_factory=VoxCPMArgs)
    piper: PiperArgs = Field(default_factory=PiperArgs)
    screen: ScreenArgs = Field(default_factory=ScreenArgs)
    negative_speech: NegativeSpeechArgs = Field(default_factory=NegativeSpeechArgs)
    augmentation: AugmentationArgs = Field(default_factory=AugmentationArgs)
    training: TrainingArgs = Field(default_factory=TrainingArgs)
    evaluation: EvaluationArgs = Field(default_factory=EvaluationArgs)

    @field_validator("wake_word")
    @classmethod
    def _normalise(cls, value: str) -> str:
        value = " ".join(value.strip().split())
        if not re.fullmatch(r"[^\W\d_][\w' -]{1,40}", value):
            raise ValueError("use 2-40 letters (spaces, hyphens and apostrophes allowed)")
        return value

    @field_validator("language")
    @classmethod
    def _supported(cls, value: str) -> str:
        if value not in LANGUAGES:
            raise ValueError(f"unsupported language {value!r}; choose from {sorted(LANGUAGES)}")
        return value

    @model_validator(mode="before")
    @classmethod
    def _native_negatives(cls, data: Any) -> Any:
        """Add negative speech in the wake-word language unless chosen explicitly."""
        if isinstance(data, dict) and data.get("language", "en") != "en":
            speech = dict(data.get("negative_speech") or {})
            speech.setdefault("language", data["language"])
            data = {**data, "negative_speech": speech}
        return data

    @classmethod
    def resolve(cls, wake_word: str, config: Path | None = None, **overrides: Any) -> Args:
        """Merge defaults, an optional YAML file and explicit overrides into validated arguments."""
        data: dict[str, Any] = {"wake_word": wake_word}
        if config is not None:
            _merge(data, yaml.safe_load(Path(config).read_text()) or {})
        _merge(data, {key: value for key, value in overrides.items() if value is not None})
        return cls.model_validate(data)

    @property
    def spec(self) -> Language:
        return LANGUAGES[self.language]

    @property
    def slug(self) -> str:
        return re.sub(r"\W+", "_", self.wake_word.lower()).strip("_")

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:10]

    def positive_phrases(self) -> list[str]:
        """Punctuation and short carriers that vary prosody around the wake word."""
        title = self.wake_word[0].upper() + self.wake_word[1:]
        phrases = [f"{title}.", f"{title}!", f"{title}?", f"{title},", f"{title}..."]
        carried = self.wake_word.lower().startswith(("hey ", "ok ", "okay "))
        if self.language == "en" and not carried:
            phrases += [f"Hey, {title}.", f"Okay {title}."]
        return phrases


def _merge(base: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
