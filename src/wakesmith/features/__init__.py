"""Augmentation and runtime-identical openWakeWord feature extraction."""

from ._augment import Augmentor, held_out, noise_sources, place, read_mono, to_int16
from ._extract import clip_features, speech_features, windows
from ._streaming import stream_embeddings

__all__ = [
    "Augmentor",
    "clip_features",
    "held_out",
    "noise_sources",
    "place",
    "read_mono",
    "speech_features",
    "stream_embeddings",
    "to_int16",
    "windows",
]
