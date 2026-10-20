"""Select the recommended threshold and rank checkpoints from validation metrics."""

from __future__ import annotations

from wakesmith.config import TrainingArgs


def _feasible(metrics: dict, args: TrainingArgs) -> bool:
    return (
        max(metrics["false_activations_per_hour"].values())
        <= args.target_false_activations_per_hour
        and metrics["adversarial_accept"] <= args.max_adversarial_accept
    )


def operating_point(
    metrics_by_threshold: dict[str, dict], args: TrainingArgs
) -> tuple[float | None, float]:
    """Return the recommended threshold and its validation recall.

    The default threshold is kept if it meets the false-activation and adversarial limits;
    otherwise the lowest higher threshold that does. Lower thresholds are never proposed:
    a single validation activation in a few hours is too little evidence to trade false
    activations for recall. Returns ``(None, 0.0)`` if no threshold qualifies.
    """
    for threshold, metrics in sorted(metrics_by_threshold.items(), key=lambda item: float(item[0])):
        if float(threshold) >= args.default_threshold and _feasible(metrics, args):
            return float(threshold), metrics["recall"]
    return None, 0.0


def objective(metrics_by_threshold: dict[str, dict], args: TrainingArgs) -> float:
    """Rank checkpoints by recall at the operating point, or by their best false-activation rate."""
    threshold, recall = operating_point(metrics_by_threshold, args)
    if threshold is not None:
        return recall
    return -min(
        max(m["false_activations_per_hour"].values()) for m in metrics_by_threshold.values()
    )
