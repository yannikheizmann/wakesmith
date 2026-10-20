"""Hold training pools on the GPU and stream ACAV100M negatives from disk."""

from __future__ import annotations

import queue
import threading
from pathlib import Path

import numpy as np
import torch

from wakesmith.config import Args
from wakesmith.features import windows

ACAV_VALIDATION_FRACTION = 0.05
"""~100 h of ACAV100M held out from training: a few hours of streams cannot resolve 0.2/h."""


class TrainingData:
    """Mixed batches from positive, confusable, noise, speech and mined pools; validation sets."""

    def __init__(self, features: Path, acav: Path, args: Args, device: torch.device):
        self.device = device
        self.batch = args.training.batch
        self.rng = np.random.default_rng(args.seed)

        def load(pattern: str) -> torch.Tensor | None:
            found = sorted(features.glob(f"{pattern}.npy"))
            return (
                torch.from_numpy(np.concatenate([np.load(f) for f in found])).to(device)
                if found
                else None
            )

        self.positive = {
            e: x for e in ("kokoro", "voxcpm") if (x := load(f"{e}_positive_train")) is not None
        }
        self.adversarial = load("*_adversarial_train")
        self.background = load("background_train")
        self.val_positive = load("*_positive_validation")
        self.val_adversarial = load("*_adversarial_validation")
        self.speech = None
        if (features / "speech_train.npz").exists():
            speech = np.load(features / "speech_train.npz")
            self.speech = torch.from_numpy(
                windows(speech["frames"], speech["offsets"], stride=4)
            ).to(device)
        acav_all = np.load(acav, mmap_mode="r")
        split = int(len(acav_all) * (1 - ACAV_VALIDATION_FRACTION))
        self.acav = acav_all[:split]
        self.acav_validation = torch.from_numpy(np.asarray(acav_all[split:])).to(device)
        validation = np.load(features / "owwval_validation.npy")
        self.streams = {"openwakeword_validation": (validation, np.array([0, len(validation)]))}
        if (features / "speech_validation.npz").exists():
            speech = np.load(features / "speech_validation.npz")
            self.streams["speech_validation"] = (speech["frames"], speech["offsets"])
        self.hard: torch.Tensor | None = None
        self._prefetch: queue.Queue = queue.Queue(maxsize=8)
        threading.Thread(target=self._read_acav, daemon=True).start()

    def _read_acav(self) -> None:
        n, block = self.batch.acav, 64
        while True:
            # Adjacent rows come from one recording; random 64-row blocks keep reads
            # sequential while mixing many recordings into every batch.
            starts = self.rng.integers(0, len(self.acav) - block, n // block)
            rows = np.concatenate([self.acav[s : s + block] for s in starts])
            batch = torch.from_numpy(rows)
            self._prefetch.put(batch.pin_memory() if self.device.type == "cuda" else batch)

    def _sample(self, pool: torch.Tensor, n: int) -> torch.Tensor:
        return pool[torch.randint(len(pool), (n,), device=self.device)]

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        """One training batch of float32 windows and 0/1 labels."""
        b = self.batch
        parts = [(self._sample(x, getattr(b, e)), 1.0) for e, x in self.positive.items()]
        parts += [
            (self._sample(self.adversarial, b.adversarial), 0.0),
            (self._sample(self.background, b.background), 0.0),
            (self._prefetch.get().to(self.device, non_blocking=True), 0.0),
        ]
        if self.speech is not None:
            parts.append((self._sample(self.speech, b.speech), 0.0))
        if self.hard is not None:
            parts.append((self._sample(self.hard, b.hard), 0.0))
        x = torch.cat([p for p, _ in parts]).float()
        y = torch.cat([torch.full((len(p),), label, device=self.device) for p, label in parts])
        return x, y
