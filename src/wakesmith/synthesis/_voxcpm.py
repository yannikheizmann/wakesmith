"""Synthesise speech with VoxCPM2 voice design in a separate worker process.

VoxCPM decodes one utterance at a time and peaks near 12 GB of host memory. Running it via
``python -m wakesmith.synthesis._voxcpm`` returns that memory, and the GPU, when it ends.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from ._clips import missing, split_seed, write_clip

Split = tuple[str, tuple[str, ...], list[str], int]


def work(
    clips: Path,
    model_dir: Path,
    splits: list[Split],
    seed: int,
    cfg_values: list[float],
    timesteps: list[int],
    worker: int = 0,
    workers: int = 1,
) -> None:
    import torch
    from voxcpm import VoxCPM

    model = VoxCPM.from_pretrained(str(model_dir), load_denoiser=False)
    rate = int(model.tts_model.sample_rate)
    for name, voices, phrases, n in splits:
        output = clips / name
        output.mkdir(parents=True, exist_ok=True)
        base = split_seed(seed, name)
        for i in missing(output, n, worker, workers):
            for attempt in range(5):
                rng = np.random.default_rng([base, i, attempt])
                torch.manual_seed(int(rng.integers(2**31)))
                voice, phrase = (
                    voices[rng.integers(len(voices))],
                    phrases[rng.integers(len(phrases))],
                )
                try:
                    audio = model.generate(
                        text=f"({voice}){phrase}",
                        cfg_value=float(rng.choice(cfg_values)),
                        inference_timesteps=int(rng.choice(timesteps)),
                    )
                except Exception as error:  # Rare sampler failures must not abort hours of work.
                    print(f"VoxCPM failed on {name} {i}: {error}", flush=True)
                    continue
                if write_clip(output / f"clip_{i:06d}.wav", audio, rate):
                    break


def command(
    clips: Path,
    model_dir: Path,
    splits: list[Split],
    seed: int,
    cfg_values,
    timesteps,
    worker: int,
    workers: int,
) -> list[str]:
    """Command line that runs :func:`work` in a fresh interpreter."""
    job = {
        "clips": str(clips),
        "model_dir": str(model_dir),
        "splits": splits,
        "seed": seed,
        "cfg_values": list(cfg_values),
        "timesteps": list(timesteps),
        "worker": worker,
        "workers": workers,
    }
    return [sys.executable, "-m", "wakesmith.synthesis._voxcpm", json.dumps(job)]


if __name__ == "__main__":
    job = json.loads(sys.argv[1])
    work(Path(job.pop("clips")), Path(job.pop("model_dir")), **job)
