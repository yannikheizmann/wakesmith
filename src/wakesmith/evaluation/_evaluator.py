"""Evaluate a trained model on held-out voices, noise, rooms and speech.

Positives stream through openWakeWord's own ``Model`` and :class:`DetectionGate`, as a
deployed integration would run them. Negative streams score precomputed runtime-identical
features. Nothing here influences training or threshold selection.
"""

from __future__ import annotations

from fractions import Fraction
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from wakesmith.config import (
    CHUNK_SAMPLES,
    CONTEXT_FRAMES,
    FRAME_SECONDS,
    SAMPLE_RATE,
    EvaluationArgs,
)
from wakesmith.features import Augmentor, place, read_mono, to_int16

from ._gate import DetectionGate, stream_events


def _detect(job: tuple) -> list[list[bool]]:
    model_path, paths, sources, noisy, seed, thresholds, consecutive, cooldown = job
    from openwakeword.model import Model

    model = Model(wakeword_models=[model_path], inference_framework="onnx")
    key = Path(model_path).stem
    rng = np.random.default_rng(seed)
    augment = Augmentor(*sources, rng)
    results = []
    for path in paths:
        audio = place(read_mono(path), 2 * SAMPLE_RATE, rng)
        audio = (
            augment(audio) if noisy else audio * (10 ** (-6 / 20) / (np.abs(audio).max() + 1e-8))
        )
        # One second of trailing silence lets a word ending at the clip end finish scoring.
        pcm = to_int16(np.concatenate([audio, np.zeros(SAMPLE_RATE, np.float32)]))
        model.reset()
        for _ in range(40):
            model.predict(np.zeros(CHUNK_SAMPLES, np.int16))
        scores = [
            float(model.predict(pcm[i : i + CHUNK_SAMPLES])[key])
            for i in range(0, len(pcm) - CHUNK_SAMPLES + 1, CHUNK_SAMPLES)
        ]
        detected = []
        for threshold in thresholds:
            gate = DetectionGate(threshold, consecutive, cooldown)
            # Exact frame times avoid float drift at the cooldown boundary.
            detected.append(
                any(
                    gate.update(s, Fraction(i * CHUNK_SAMPLES, SAMPLE_RATE))
                    for i, s in enumerate(scores)
                )
            )
        results.append(detected)
    return results


def recall(
    model: Path,
    clips: list[Path],
    sources,
    noisy: bool,
    seed: int,
    thresholds: list[float],
    args: EvaluationArgs,
) -> dict[str, float]:
    """Fraction of clips that activate the streaming model at each threshold."""
    paths = [str(p) for p in clips]
    jobs = [
        (
            str(model),
            paths[i : i + 25],
            sources,
            noisy,
            seed + i,
            thresholds,
            args.consecutive_frames,
            args.cooldown_ms / 1000,
        )
        for i in range(0, len(paths), 25)
    ]
    with get_context("spawn").Pool(min(24, len(jobs))) as pool:
        hits = np.array([row for part in pool.map(_detect, jobs) for row in part])
    return {str(t): float(hits[:, i].mean()) for i, t in enumerate(thresholds)}


def false_activations(
    model: Path,
    frames: np.ndarray,
    offsets: np.ndarray,
    thresholds: list[float],
    args: EvaluationArgs,
) -> tuple[float, dict[str, float]]:
    """Hours of speech and activations per hour at each threshold, scoring every 80 ms."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    scores, new_offsets = [], [0]
    for a, b in zip(offsets[:-1], offsets[1:], strict=True):
        if b - a < CONTEXT_FRAMES:
            continue
        x = np.lib.stride_tricks.sliding_window_view(
            np.asarray(frames[a:b], np.float32), CONTEXT_FRAMES, axis=0
        )
        x = x.transpose(0, 2, 1)
        for i in range(0, len(x), 4096):
            scores.append(
                session.run(None, {name: np.ascontiguousarray(x[i : i + 4096])})[0].reshape(-1)
            )
        new_offsets.append(new_offsets[-1] + len(x))
    scores = np.concatenate(scores)
    hours = len(scores) * FRAME_SECONDS / 3600
    gate = {
        "consecutive": args.consecutive_frames,
        "cooldown_frames": round(args.cooldown_ms / 1000 / FRAME_SECONDS),
    }
    return hours, {
        str(t): stream_events(scores, np.array(new_offsets), t, **gate) / hours for t in thresholds
    }
