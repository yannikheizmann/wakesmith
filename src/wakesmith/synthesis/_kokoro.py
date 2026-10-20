"""Synthesise speech with Kokoro-82M from random blends of its voice embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._clips import missing, write_clip


class Kokoro:
    """Kokoro-82M on the GPU with English or espeak grapheme-to-phoneme front ends."""

    def __init__(self, root: Path, espeak: str | None = None):
        from kokoro import KModel

        self.root, self.espeak = root, espeak
        self.model = (
            KModel(
                repo_id="hexgrad/Kokoro-82M",
                config=str(root / "config.json"),
                model=str(root / "kokoro-v1_0.pth"),
            )
            .to("cuda")
            .eval()
        )
        self._pipelines: dict[str, object] = {}
        self._packs: dict[str, object] = {}

    def _pack(self, name: str):
        import torch

        if name not in self._packs:  # CPU FloatTensors, as KPipeline expects.
            self._packs[name] = torch.load(
                self.root / "voices" / f"{name}.pt", weights_only=True
            ).float()
        return self._packs[name]

    def _pipeline(self, code: str):
        from kokoro import KPipeline

        if code not in self._pipelines:
            if code.startswith("espeak:"):
                from misaki.espeak import EspeakG2P

                pipeline = KPipeline(lang_code="e", repo_id="hexgrad/Kokoro-82M", model=False)
                pipeline.g2p = EspeakG2P(language=code.removeprefix("espeak:"))
            else:
                pipeline = KPipeline(lang_code=code, repo_id="hexgrad/Kokoro-82M", model=False)
            self._pipelines[code] = pipeline
        return self._pipelines[code]

    def generate(
        self,
        output: Path,
        phrases: list[str],
        n: int,
        seed: int,
        voices: tuple[str, ...],
        front_ends: dict[str, float],
        speed: tuple[float, float],
        blend: bool = True,
    ) -> None:
        """Write ``n`` clips; each speaker is a convex mix of two or three voices if ``blend``."""
        import torch

        output.mkdir(parents=True, exist_ok=True)
        codes = list(front_ends)
        weights = np.asarray(list(front_ends.values())) / sum(front_ends.values())
        for i in missing(output, n):
            for attempt in range(5):
                rng = np.random.default_rng([seed, i, attempt])
                if blend:
                    chosen = rng.choice(voices, size=int(rng.integers(2, 4)), replace=False)
                    mix = rng.dirichlet(np.ones(len(chosen)) * 0.7)
                    voice = sum(
                        float(w) * self._pack(str(v)) for w, v in zip(mix, chosen, strict=True)
                    )
                else:
                    voice = self._pack(str(rng.choice(voices)))
                pipeline = self._pipeline(str(rng.choice(codes, p=weights)))
                audio = []
                with torch.no_grad():
                    for result in pipeline(
                        str(rng.choice(phrases)),
                        voice=voice,
                        model=self.model,
                        speed=float(rng.uniform(*speed)),
                    ):
                        if result.audio is not None:
                            audio.append(result.audio.cpu().numpy())
                if audio and write_clip(output / f"clip_{i:06d}.wav", np.concatenate(audio), 24000):
                    break
