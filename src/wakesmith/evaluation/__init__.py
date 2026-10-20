"""Activation gating, threshold selection and held-out evaluation."""

from ._calibration import objective, operating_point
from ._evaluator import false_activations, recall
from ._gate import DetectionGate, count_events, stream_events
from ._report import model_card, report

__all__ = [
    "DetectionGate",
    "count_events",
    "false_activations",
    "model_card",
    "objective",
    "operating_point",
    "recall",
    "report",
    "stream_events",
]
