"""Synthesise English speech with Piper LibriTTS through livekit-wakeword's wrapper.

Two upstream behaviours are replaced: the wrapper shells out to a system ``espeak-ng``
(the bundled library is used instead), and it deletes every VAD-silent frame inside a
clip, which cuts stop closures and weak onsets (only the edges are trimmed here).
"""

from __future__ import annotations

import functools
import random
from pathlib import Path

import numpy as np

from wakesmith.config import PiperArgs

from ._clips import count, trim_edges


@functools.cache
def _patch() -> None:
    from livekit.wakeword.data.piper import synthesis

    from wakesmith.phonetics import phonemize

    synthesis._find_espeak_ng = lambda: "espeakng-loader"
    synthesis._espeak_phonemize = lambda text, voice="en-us": phonemize([text], voice)[0]
    synthesis.remove_silence = trim_edges


def generate(
    model: Path, output: Path, phrases: list[str], n: int, seed: int, args: PiperArgs
) -> None:
    """Write ``n`` clips, resuming from the clips already present."""
    existing = count(output)
    if existing >= n:
        return
    import torch
    from livekit.wakeword.data.piper import generate_samples

    _patch()
    random.seed(seed + existing)
    np.random.seed(seed + existing)
    torch.manual_seed(seed + existing)
    generate_samples(
        text=phrases,
        output_dir=output,
        max_samples=n,
        model=model,
        batch_size=args.batch_size,
        slerp_weights=list(args.slerp_weights),
        length_scales=list(args.length_scales),
        noise_scales=list(args.noise_scales),
        noise_scale_ws=list(args.noise_scale_ws),
        start_index=existing,
    )
