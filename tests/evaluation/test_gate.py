from fractions import Fraction

import numpy as np

from wakesmith.evaluation import DetectionGate, count_events, stream_events


def test_vectorised_count_matches_the_streaming_gate():
    rng = np.random.default_rng(1)
    for _ in range(200):
        scores = rng.random(rng.integers(1, 400)) ** rng.uniform(0.2, 4)
        for threshold in (0.3, 0.5, 0.9):
            gate = DetectionGate(threshold, 2, 2.0)
            expected = sum(gate.update(float(s), Fraction(2 * i, 25)) for i, s in enumerate(scores))
            assert count_events(scores, threshold) == expected


def test_cooldown_does_not_span_recordings():
    scores = np.array([1.0, 1.0, 1.0, 1.0])
    assert count_events(scores, 0.5) == 1
    assert stream_events(scores, np.array([0, 2, 4]), 0.5) == 2
