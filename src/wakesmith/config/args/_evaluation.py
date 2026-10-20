"""Define held-out evaluation arguments."""

from __future__ import annotations

from pydantic import Field, PositiveInt

from ._base import ArgsBase


class EvaluationArgs(ArgsBase):
    """Activation gate and reported thresholds for the held-out evaluation."""

    thresholds: tuple[float, ...] = (0.3, 0.5, 0.7, 0.9)
    consecutive_frames: PositiveInt = Field(default=2, description="Frames at or above threshold.")
    cooldown_ms: PositiveInt = Field(default=2000, description="Minimum time between activations.")
