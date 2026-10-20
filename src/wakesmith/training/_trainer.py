"""Train the classifier with validation-driven checkpoint selection and hard-negative mining.

The head and loss come from livekit-wakeword: a temporal conv-attention classifier trained
with focal loss, embedding mixup, label smoothing, three learning-rate phases and averaging
of the best checkpoints. Validation streams continuous speech through the activation gate,
so false activations are measured per real hour, and after the first phase the windows of
all negative speech that score highest are mined and oversampled.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

from wakesmith.config import CONTEXT_FRAMES, EMBEDDING_DIM, FRAME_SECONDS, Args
from wakesmith.evaluation import objective, operating_point, stream_events

from ._data import TrainingData

THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)
Progress = Callable[[int, int, str], None]


@dataclass
class TrainingResult:
    """Averaged model, its validation metrics, the recommended threshold and the run log."""

    model: torch.nn.Module
    metrics: dict
    threshold: float | None
    log: list[dict] = field(default_factory=list)


@torch.no_grad()
def _predict(model: torch.nn.Module, x: torch.Tensor, batch: int = 8192) -> np.ndarray:
    model.eval()
    out = [model(x[i : i + batch].float()).squeeze(-1).cpu() for i in range(0, len(x), batch)]
    return torch.cat(out).numpy() if out else np.zeros(0)


@torch.no_grad()
def _stream_scores(
    model: torch.nn.Module, frames: np.ndarray, offsets: np.ndarray, device
) -> tuple:
    """Score every frame position, as a streaming runtime does every 80 ms."""
    frames_gpu = torch.from_numpy(np.asarray(frames, dtype=np.float16)).to(device)
    index = torch.arange(CONTEXT_FRAMES, device=device)
    scores, new_offsets = [], [0]
    for a, b in zip(offsets[:-1], offsets[1:], strict=True):
        if b - a < CONTEXT_FRAMES:
            continue
        starts = torch.arange(a, b - CONTEXT_FRAMES + 1, device=device)
        for i in range(0, len(starts), 16384):
            s = starts[i : i + 16384]
            scores.append(model(frames_gpu[s[:, None] + index].float()).squeeze(-1).cpu())
        new_offsets.append(new_offsets[-1] + len(starts))
    return torch.cat(scores).numpy(), np.array(new_offsets)


def validate(model: torch.nn.Module, data: TrainingData) -> dict[str, dict]:
    """Recall, adversarial accepts and false activations per hour at each threshold."""
    positive = _predict(model, data.val_positive)
    adversarial = _predict(model, data.val_adversarial)
    streams = {
        name: _stream_scores(model, f, o, data.device) for name, (f, o) in data.streams.items()
    }
    acav = _predict(model, data.acav_validation)
    acav_hours = len(acav) * CONTEXT_FRAMES * FRAME_SECONDS / 3600
    metrics = {}
    for t in THRESHOLDS:
        rates = {
            name: stream_events(s, o, t) / (len(s) * FRAME_SECONDS / 3600)
            for name, (s, o) in streams.items()
        }
        # Independent windows with one score each: a lower bound on streaming activations.
        rates["acav_heldout_windows"] = float((acav >= t).sum()) / acav_hours
        metrics[str(t)] = {
            "recall": float((positive >= t).mean()),
            "adversarial_accept": float((adversarial >= t).mean()),
            "false_activations_per_hour": rates,
        }
    model.train()
    return metrics


@torch.no_grad()
def mine_hard_negatives(
    model: torch.nn.Module, data: TrainingData, n: int, chunk: int = 262144
) -> dict:
    """Keep the ``n`` highest-scoring windows of all negative speech for oversampling.

    Random batches rarely revisit the few moments of speech that resemble the wake word.
    """
    model.eval()
    best_scores = torch.empty(0, device=data.device)
    best = torch.empty(0, CONTEXT_FRAMES, EMBEDDING_DIM, dtype=torch.float16, device=data.device)
    sources = [data.acav[i : i + chunk] for i in range(0, len(data.acav), chunk)]
    if data.speech is not None:
        sources += [data.speech[i : i + chunk] for i in range(0, len(data.speech), chunk)]
    for rows in sources:
        x = (
            (rows if torch.is_tensor(rows) else torch.from_numpy(np.asarray(rows)))
            .to(data.device)
            .half()
        )
        scores = torch.cat(
            [model(x[i : i + 16384].float()).squeeze(-1) for i in range(0, len(x), 16384)]
        )
        best_scores, best = torch.cat([best_scores, scores]), torch.cat([best, x])
        keep = torch.topk(best_scores, min(n, len(best_scores))).indices
        best_scores, best = best_scores[keep], best[keep]
    model.train()
    data.hard = best
    return {
        "count": len(best),
        "min_score": float(best_scores.min()),
        "max_score": float(best_scores.max()),
    }


def build_model(args: Args) -> torch.nn.Module:
    """livekit-wakeword's conv-attention classifier for ``(batch, 16, 96)`` embeddings."""
    from livekit.wakeword.config import WakeWordConfig
    from livekit.wakeword.models.pipeline import WakeWordClassifier

    config = WakeWordConfig(
        model_name=args.slug,
        target_phrases=[args.wake_word],
        model={"model_type": "conv_attention", "model_size": args.training.model_size},
    )
    return WakeWordClassifier(config)


def train(args: Args, data: TrainingData, progress: Progress | None = None) -> TrainingResult:
    """Run three phases, mine hard negatives after the first and average the best checkpoints."""
    from livekit.wakeword.training.trainer import (
        _cosine_warmup_schedule,
        _negative_weight_schedule,
        focal_loss,
    )

    training = args.training
    torch.manual_seed(args.seed)
    model = build_model(args).to(data.device)
    phases = [
        (training.steps, training.learning_rate),
        (training.steps // 10, training.learning_rate * 0.1),
        (training.steps // 10, training.learning_rate * 0.01),
    ]
    total, done = sum(steps for steps, _ in phases), 0
    checkpoints, log, max_weight = [], [], training.max_negative_weight
    for phase, (steps, lr) in enumerate(phases, 1):
        if phase == 2 and training.hard_negatives:
            if progress:
                progress(done, total, "mining hard negatives")
            log.append(
                {"hard_negative_mining": mine_hard_negatives(model, data, training.hard_negatives)}
            )
        if phase > 1:
            rates = validate(model, data)[str(training.default_threshold)][
                "false_activations_per_hour"
            ]
            if max(rates.values()) > training.target_false_activations_per_hour:
                max_weight *= 2
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=training.weight_decay)
        warmup, hold = (steps // 5, steps // 3) if phase == 1 else (0, 0)
        interval = max(1, steps // 20)
        model.train()
        for step in range(steps):
            x, y = data.next()
            lam = (
                torch.distributions.Beta(training.mixup_alpha, training.mixup_alpha)
                .sample()
                .to(data.device)
            )
            perm = torch.randperm(len(x), device=data.device)
            x, y = lam * x + (1 - lam) * x[perm], lam * y + (1 - lam) * y[perm]
            y = y * (1 - training.label_smoothing) + 0.5 * training.label_smoothing
            weight = torch.where(y < 0.5, _negative_weight_schedule(step, steps, max_weight), 1.0)
            loss = (focal_loss(model(x).squeeze(-1), y, gamma=training.focal_gamma) * weight).mean()
            for group in optimizer.param_groups:
                group["lr"] = _cosine_warmup_schedule(step, steps, warmup, hold, lr)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            done += 1
            if progress and done % 200 == 0:
                progress(done, total, f"phase {phase}/3")
            if step >= steps * 3 // 4 and (step + 1) % interval == 0:
                metrics = validate(model, data)
                score = objective(metrics, training)
                checkpoints.append(
                    (score, {k: v.detach().clone() for k, v in model.state_dict().items()})
                )
                log.append({"phase": phase, "step": step, "objective": score, "metrics": metrics})
    best = sorted(checkpoints, key=lambda c: c[0], reverse=True)[:5]
    model.load_state_dict(
        {k: torch.stack([c[1][k].float() for c in best]).mean(0) for k in best[0][1]}
    )
    metrics = validate(model, data)
    threshold, recall_ = operating_point(metrics, training)
    log.append(
        {
            "averaged_checkpoints": len(best),
            "metrics": metrics,
            "threshold": threshold,
            "recall": recall_,
        }
    )
    return TrainingResult(model, metrics, threshold, log)
