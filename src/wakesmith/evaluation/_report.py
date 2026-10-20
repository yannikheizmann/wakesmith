"""Write the model card and a human-readable report for a trained model."""

from __future__ import annotations

from pathlib import Path

from wakesmith import __version__
from wakesmith.config import Args


def model_card(
    args: Args, sha256: str, threshold: float, evaluation: dict, validation: dict
) -> dict:
    """Machine-readable metadata stored next to the ONNX model."""
    key = str(threshold)
    return {
        "wake_word": args.wake_word,
        "language": args.language,
        "sha256": sha256,
        "format": "openWakeWord classifier: (batch, 16, 96) embeddings to (batch, 1) score",
        "architecture": f"livekit-wakeword conv-attention head, {args.training.model_size}",
        "recommended": {
            "threshold": threshold,
            "consecutive_frames": args.evaluation.consecutive_frames,
            "cooldown_ms": args.evaluation.cooldown_ms,
        },
        "held_out": {
            "recall": {name: values[key] for name, values in evaluation["recall"].items()},
            "false_activations_per_hour": {
                name: s["false_activations_per_hour"][key]
                for name, s in evaluation["negative_streams"].items()
            },
            "negative_hours": {
                name: round(s["hours"], 1) for name, s in evaluation["negative_streams"].items()
            },
        },
        "validation_at_threshold": validation.get(key),
        "args_digest": args.digest,
        "wakesmith_version": __version__,
        "limitations": [
            "Positives are synthetic; recordings of real users remain the strongest next step.",
            "Negative streams are read or web speech, not continuous room audio with TV and music.",
            "Measurements are samples, not statistical guarantees.",
        ],
    }


def report(card: dict, evaluation: dict, screen: dict, path: Path) -> None:
    """Markdown summary of what was trained and how it measured."""
    threshold = card["recommended"]["threshold"]
    lines = [
        f"# {card['wake_word']} — wakesmith model report",
        "",
        f"- Language: `{card['language']}`",
        f"- Recommended threshold: **{threshold}** "
        f"({card['recommended']['consecutive_frames']} consecutive frames, "
        f"{card['recommended']['cooldown_ms']} ms cooldown)",
        f"- Model SHA-256: `{card['sha256']}`",
        "",
        "## Held-out recall",
        "",
        "| Set | " + " | ".join(evaluation["thresholds"]) + " |",
        "|---|" + "---:|" * len(evaluation["thresholds"]),
    ]
    for name, values in evaluation["recall"].items():
        lines.append(
            f"| {name} | " + " | ".join(f"{values[t]:.1%}" for t in evaluation["thresholds"]) + " |"
        )
    lines += [
        "",
        "## Held-out false activations per hour",
        "",
        "| Speech | Hours | " + " | ".join(evaluation["thresholds"]) + " |",
        "|---|---:|" + "---:|" * len(evaluation["thresholds"]),
    ]
    for name, stream in evaluation["negative_streams"].items():
        rates = stream["false_activations_per_hour"]
        lines.append(
            f"| {name} | {stream['hours']:.1f} | "
            + " | ".join(f"{rates[t]:.2f}" for t in evaluation["thresholds"])
            + " |"
        )
    lines += [
        "",
        "## Synthetic data after the ASR screen",
        "",
        "| Split | Generated | Accepted |",
        "|---|---:|---:|",
    ]
    lines += [f"| {name} | {s['generated']} | {s['accepted']} |" for name, s in screen.items()]
    lines += ["", "## Limitations", "", *[f"- {item}" for item in card["limitations"]], ""]
    path.write_text("\n".join(lines))
