"""Screen synthetic clips with Whisper and the phonetic matcher.

Text-to-speech regularly mispronounces invented words. A positive that does not sound like
the wake word teaches the detector the wrong thing, so every positive must be transcribed
as the wake word; adversarial clips transcribed as the wake word are dropped.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from wakesmith.config import TRANSCRIPTS_FILENAME
from wakesmith.phonetics import Matcher

from ._clips import clips


class Whisper:
    """Batched half-precision Whisper transcription on the GPU."""

    def __init__(self, model: str, revision: str, language: str, batch_size: int):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.processor = WhisperProcessor.from_pretrained(model, revision=revision)
        self.model = (
            WhisperForConditionalGeneration.from_pretrained(
                model, revision=revision, torch_dtype=torch.float16
            )
            .to("cuda")
            .eval()
        )
        self.language, self.batch_size = language, batch_size

    def __call__(self, audio: list) -> list[str]:
        import torch

        inputs = self.processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt", device="cuda"
        )
        with torch.no_grad():
            tokens = self.model.generate(
                inputs.input_features.to("cuda", torch.float16),
                language=self.language,
                task="transcribe",
                max_new_tokens=24,
            )
        return [
            text.strip() for text in self.processor.batch_decode(tokens, skip_special_tokens=True)
        ]


def transcribe(directory: Path, asr: Whisper, advance: Callable[[int], None] | None = None) -> None:
    """Transcribe every clip of a split once, caching results next to the clips."""
    import soundfile as sf

    cache = directory / TRANSCRIPTS_FILENAME
    transcripts = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [p for p in clips(directory) if p.name not in transcripts]
    if advance:
        advance(len(transcripts))
    for start in range(0, len(todo), asr.batch_size):
        batch = todo[start : start + asr.batch_size]
        texts = asr([sf.read(p, dtype="float32")[0] for p in batch])
        transcripts.update({p.name: t for p, t in zip(batch, texts, strict=True)})
        cache.write_text(json.dumps(transcripts, indent=0))
        if advance:
            advance(len(batch))


def accepted(directory: Path, matcher: Matcher) -> list[Path]:
    """Positives heard as the wake word; negatives not heard as it, or all if unscreened."""
    positive = "_positive_" in directory.name
    cache = directory / TRANSCRIPTS_FILENAME
    if not cache.exists():
        return [] if positive else clips(directory)
    transcripts = json.loads(cache.read_text())
    paths = clips(directory)
    heard = matcher([transcripts.get(p.name, "") for p in paths])
    return [p for p, h in zip(paths, heard, strict=True) if h == positive]
