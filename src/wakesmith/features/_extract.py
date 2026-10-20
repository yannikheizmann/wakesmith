"""Extract feature datasets from clips and continuous speech in worker processes."""

from __future__ import annotations

import os

# One worker per core: keep BLAS and ONNX single-threaded inside each worker.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import io  # noqa: E402
from collections.abc import Callable, Iterator  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from wakesmith.config import CHUNK_SAMPLES, CLIP_SECONDS, CONTEXT_FRAMES, SAMPLE_RATE  # noqa: E402

from ._augment import Augmentor, place, read_mono, to_int16  # noqa: E402
from ._streaming import stream_embeddings  # noqa: E402

Sources = tuple[list[str], list[str]]


def _workers() -> int:
    return max(1, (os.cpu_count() or 2) - 2)


def _clip_job(args: tuple) -> np.ndarray:
    paths, seed, sources, clean, background = args
    rng = np.random.default_rng(seed)
    augment = Augmentor(*sources, rng)
    length = int(CLIP_SECONDS * SAMPLE_RATE)
    windows_ = []
    for path in paths:
        if background:  # Pure noise, music and babble are standalone negatives.
            audio = augment.background(length)
            audio = audio if clean else augment.level(augment.reverberate(audio))
        else:
            audio = place(read_mono(path), length, rng)
            audio = audio if clean else augment(audio)
        windows_.append(
            stream_embeddings(to_int16(audio), int(rng.integers(0, CHUNK_SAMPLES)))[
                -CONTEXT_FRAMES:
            ]
        )
    return np.stack(windows_).astype(np.float16)


def clip_features(
    paths: list,
    output: Path,
    sources: Sources,
    rounds: int,
    seed: int,
    advance: Callable[[int], None] | None = None,
    clean: bool = False,
    background: bool = False,
) -> None:
    """Save ``(len(paths) * rounds, 16, 96)`` float16 windows, one per augmented clip."""
    if output.exists():
        return
    paths = [str(p) for p in paths]
    jobs = [
        (paths[i : i + 64], seed + r * 1_000_003 + i, sources, clean, background)
        for r in range(rounds)
        for i in range(0, len(paths), 64)
    ]
    parts = []
    with get_context("spawn").Pool(_workers()) as pool:
        for part in pool.imap(_clip_job, jobs):
            parts.append(part)
            if advance:
                advance(len(part))
    _save(output, np.concatenate(parts))


def _speech_job(args: tuple) -> list[np.ndarray]:
    recordings, seed, sources, augment = args
    import soundfile as sf

    rng = np.random.default_rng(seed)
    aug = Augmentor(*sources, rng, snr=(0.0, 25.0)) if augment else None
    streams = []
    for data in recordings:
        audio, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        audio = audio[:, 0]
        if rate != SAMPLE_RATE:
            from scipy.signal import resample_poly

            audio = resample_poly(audio, SAMPLE_RATE, rate).astype(np.float32)
        if aug and rng.random() < 0.5:
            audio = aug.level(aug.add_background(aug.reverberate(audio), p=0.5))
        streams.append(
            stream_embeddings(to_int16(audio), int(rng.integers(0, CHUNK_SAMPLES))).astype(
                np.float16
            )
        )
    return streams


def _recordings(shards: list[Path], hours: float) -> Iterator[tuple[bytes, float]]:
    import pyarrow.parquet as pq

    seconds = 0.0
    for shard in shards:
        table = pq.ParquetFile(shard)
        for group in range(table.num_row_groups):
            for row in table.read_row_group(group, columns=["audio", "audio_duration"]).to_pylist():
                yield row["audio"]["bytes"], row["audio_duration"]
                seconds += row["audio_duration"]
                if seconds >= hours * 3600:
                    return


def speech_features(
    shards: list[Path],
    output: Path,
    hours: float,
    sources: Sources,
    seed: int,
    augment: bool,
    advance: Callable[[float], None] | None = None,
) -> None:
    """Save continuous-speech frame streams as ``frames`` with per-recording ``offsets``."""
    if output.exists():
        return
    durations: list[float] = []

    def jobs() -> Iterator[tuple]:
        batch, seconds = [], 0.0
        for data, duration in _recordings(shards, hours):
            batch.append(data)
            seconds += duration
            if len(batch) == 16:
                durations.append(seconds)
                yield batch, seed + len(durations), sources, augment
                batch, seconds = [], 0.0
        if batch:
            durations.append(seconds)
            yield batch, seed + len(durations), sources, augment

    streams = []
    with get_context("spawn").Pool(_workers()) as pool:
        for index, result in enumerate(pool.imap(_speech_job, jobs())):
            streams.extend(result)
            if advance:
                advance(durations[index] / 3600)
    _save(
        output, frames=np.concatenate(streams), offsets=np.cumsum([0] + [len(s) for s in streams])
    )


def windows(frames: np.ndarray, offsets: np.ndarray, stride: int) -> np.ndarray:
    """Cut sliding 16-frame windows that never cross a recording boundary."""
    starts = np.concatenate(
        [
            np.arange(a, b - CONTEXT_FRAMES + 1, stride)
            for a, b in zip(offsets[:-1], offsets[1:], strict=True)
            if b - a >= CONTEXT_FRAMES
        ]
    )
    return frames[starts[:, None] + np.arange(CONTEXT_FRAMES)]


def _save(output: Path, array: np.ndarray | None = None, **arrays: np.ndarray) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.stem + ".partial" + output.suffix)
    if array is not None:
        np.save(partial, array)
    else:
        np.savez(partial, **arrays)
    os.replace(partial, output)
