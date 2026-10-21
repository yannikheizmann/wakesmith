"""Define classifier, optimisation and model-selection arguments."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, NonNegativeInt, PositiveFloat, PositiveInt

from ._base import ArgsBase


class BatchArgs(ArgsBase):
    """Examples drawn from each pool for one optimiser step."""

    kokoro: NonNegativeInt = 80
    voxcpm: NonNegativeInt = 48
    adversarial: NonNegativeInt = 128
    acav: PositiveInt = Field(default=1024, multiple_of=64)
    speech: NonNegativeInt = 256
    background: NonNegativeInt = 64
    hard: NonNegativeInt = 128


class TrainingArgs(ArgsBase):
    """livekit-wakeword conv-attention training with hard-negative mining."""

    model_size: Literal["tiny", "small", "medium", "large"] = "medium"
    steps: PositiveInt = Field(
        default=60000, description="Phase-one steps; phases two and three add 10% each."
    )
    learning_rate: PositiveFloat = 1e-4
    weight_decay: float = Field(default=0.01, ge=0)
    label_smoothing: float = Field(default=0.05, ge=0, lt=0.5)
    mixup_alpha: PositiveFloat = 0.2
    focal_gamma: float = Field(default=2.0, ge=0)
    max_negative_weight: PositiveFloat = 1500
    hard_negatives: NonNegativeInt = Field(
        default=20000, description="Mined windows; 0 disables mining."
    )
    target_false_activations_per_hour: PositiveFloat = 0.2
    max_adversarial_accept: float = Field(default=0.01, ge=0, le=1)
    default_threshold: float = Field(
        default=0.5,
        gt=0,
        lt=1,
        description="Recommended unless validation requires a higher threshold.",
    )
    micro: bool = Field(
        default=False,
        description=(
            "Train a microWakeWord model (int8 TFLite + ESPHome manifest) for "
            "ESP32-S3/Voice PE devices instead of an openWakeWord ONNX model. "
            "Requires the `micro` extra and a CPU/GPU TensorFlow environment."
        ),
    )
    batch: BatchArgs = Field(default_factory=BatchArgs)
