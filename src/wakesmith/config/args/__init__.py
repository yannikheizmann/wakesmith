"""Typed argument groups and the root ``Args`` that composes them."""

from ._args import Args
from ._data import AugmentationArgs, NegativeSpeechArgs
from ._evaluation import EvaluationArgs
from ._synthesis import KokoroArgs, PiperArgs, ScreenArgs, VoxCPMArgs
from ._training import BatchArgs, TrainingArgs

__all__ = [
    "Args",
    "AugmentationArgs",
    "BatchArgs",
    "EvaluationArgs",
    "KokoroArgs",
    "NegativeSpeechArgs",
    "PiperArgs",
    "ScreenArgs",
    "TrainingArgs",
    "VoxCPMArgs",
]
