"""Synthetic speech from Kokoro, VoxCPM2 and Piper, screened by Whisper."""

from ._clips import clips, count, write_clip
from ._screen import Whisper, accepted
from ._synthesiser import Split, Synthesiser

__all__ = ["Split", "Synthesiser", "Whisper", "accepted", "clips", "count", "write_clip"]
