"""Count activations with the semantics of a streaming detection gate."""

from __future__ import annotations

import numpy as np


class DetectionGate:
    """Activate after ``consecutive`` frames at or above ``threshold``, then wait ``cooldown`` s.

    This is the gate used by typical openWakeWord integrations: after an activation, the next
    one needs a fresh run of hits and the cooldown to have elapsed.
    """

    def __init__(self, threshold: float, consecutive: int = 2, cooldown: float = 2.0):
        self.threshold, self.consecutive, self.cooldown = threshold, consecutive, cooldown
        self.hits, self.last = 0, -float("inf")

    def update(self, score: float, now: float) -> bool:
        self.hits = self.hits + 1 if score >= self.threshold else 0
        if self.hits >= self.consecutive and now - self.last >= self.cooldown:
            self.last, self.hits = now, 0
            return True
        return False


def count_events(
    scores: np.ndarray, threshold: float, consecutive: int = 2, cooldown_frames: int = 25
) -> int:
    """Count :class:`DetectionGate` activations over one 80 ms frame per score, vectorised.

    Exact whenever ``cooldown_frames >= consecutive``: a frame activates if it ends a run of
    ``consecutive`` hits and the cooldown since the previous activation has elapsed.
    """
    above = np.asarray(scores) >= threshold
    run = above.copy()
    for k in range(1, consecutive):
        run[k:] &= above[:-k]
        run[:k] = False
    events, last = 0, -(10**9)
    for index in np.flatnonzero(run):
        if index - last >= cooldown_frames:
            events, last = events + 1, index
    return events


def stream_events(scores: np.ndarray, offsets: np.ndarray, threshold: float, **gate: int) -> int:
    """Count activations per recording so that a cooldown never spans unrelated audio."""
    return sum(
        count_events(scores[a:b], threshold, **gate)
        for a, b in zip(offsets[:-1], offsets[1:], strict=True)
    )
