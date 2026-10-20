"""Plan, synthesise and screen every split of a run."""

from __future__ import annotations

import gc
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from wakesmith.config import Args
from wakesmith.data import Assets
from wakesmith.phonetics import Matcher, confusables

from . import _piper, _voxcpm
from ._clips import count, split_seed
from ._kokoro import Kokoro
from ._screen import Whisper, accepted, transcribe

Progress = Callable[[int, int], None]


@dataclass(frozen=True)
class Split:
    """One directory of clips from one engine."""

    name: str
    engine: str
    phrases: tuple[str, ...]
    size: int
    voices: tuple[str, ...] = ()
    blend: bool = True

    @property
    def positive(self) -> bool:
        return "_positive_" in self.name


class Synthesiser:
    """Synthetic positives and confusable negatives for one wake word."""

    def __init__(self, args: Args, assets: Assets, root: Path):
        self.args, self.assets, self.root = args, assets, root
        spec = args.spec
        self.matcher = Matcher(args.wake_word, spec.espeak, args.screen.vowel_tolerance)
        self.negatives = confusables(
            args.wake_word, args.extra_negative_phrases, args.language, spec.espeak
        )

    def splits(self) -> list[Split]:
        r, k, v, p = self.args, self.args.kokoro, self.args.voxcpm, self.args.piper
        positive = tuple(r.positive_phrases())
        spoken = tuple(f"{n[0].upper()}{n[1:]}." for n in self.negatives)
        voices = tuple(x for x in k.voices if x not in k.evaluation_voices)
        plan = [
            Split("kokoro_positive_train", "kokoro", positive, k.n_train, voices),
            Split("kokoro_positive_validation", "kokoro", positive, k.n_validation, voices),
            Split(
                "kokoro_positive_evaluation",
                "kokoro",
                positive,
                k.n_evaluation,
                k.evaluation_voices,
                False,
            ),
            Split("kokoro_adversarial_train", "kokoro", spoken, k.n_adversarial_train, voices),
            Split(
                "kokoro_adversarial_validation",
                "kokoro",
                spoken,
                k.n_adversarial_validation,
                voices,
            ),
        ]
        if v.enabled:
            unique = tuple(dict.fromkeys(spoken))[:80]
            plan += [
                Split("voxcpm_positive_train", "voxcpm", positive, v.n_train, v.train_voices),
                Split(
                    "voxcpm_positive_validation", "voxcpm", positive, v.n_validation, v.train_voices
                ),
                Split(
                    "voxcpm_positive_evaluation",
                    "voxcpm",
                    positive,
                    v.n_evaluation,
                    v.evaluation_voices,
                ),
                Split(
                    "voxcpm_adversarial_train",
                    "voxcpm",
                    unique,
                    v.n_adversarial_train,
                    v.train_voices,
                ),
                Split(
                    "voxcpm_adversarial_validation",
                    "voxcpm",
                    unique,
                    v.n_adversarial_validation,
                    v.train_voices,
                ),
            ]
        if r.language == "en":  # Piper's LibriTTS voice is English only.
            plan += [
                Split(
                    "piper_adversarial_train", "piper", tuple(self.negatives), p.n_adversarial_train
                ),
                Split(
                    "piper_adversarial_validation",
                    "piper",
                    tuple(self.negatives),
                    p.n_adversarial_validation,
                ),
            ]
        return [s for s in plan if s.size]

    def synthesise(self, progress: Progress, status: Callable[[str], None]) -> None:
        """Generate every missing clip, engine by engine."""
        plan = self.splits()
        total = sum(s.size for s in plan)
        with _Monitor([self.root / s.name for s in plan], total, progress):
            self._kokoro([s for s in plan if s.engine == "kokoro"], status)
            self._piper([s for s in plan if s.engine == "piper"], status)
            self._voxcpm([s for s in plan if s.engine == "voxcpm"], status)

    def screen(self, progress: Progress) -> dict[str, dict[str, int]]:
        """Transcribe positives and synthetic-voice negatives; report accepted counts.

        Piper negatives are not screened: its voice rarely produces an intelligible wake
        word even when asked to.
        """
        plan = [s for s in self.splits() if s.engine != "piper"]
        total, done = sum(count(self.root / s.name) for s in plan), 0
        asr = Whisper(
            self.args.screen.model,
            self.args.screen.revision,
            self.args.spec.whisper,
            self.args.screen.batch_size,
        )

        def advance(n: int) -> None:
            nonlocal done
            done += n
            progress(done, total)

        for split in plan:
            transcribe(self.root / split.name, asr, advance)
        del asr
        _free_gpu()
        return {
            s.name: {"generated": count(self.root / s.name), "accepted": len(self.accepted(s.name))}
            for s in self.splits()
        }

    def accepted(self, name: str) -> list[Path]:
        return accepted(self.root / name, self.matcher)

    def _kokoro(self, plan: list[Split], status: Callable[[str], None]) -> None:
        todo = [s for s in plan if count(self.root / s.name) < s.size]
        if not todo:
            return
        status("Kokoro")
        spec, k = self.args.spec, self.args.kokoro
        if self.args.language == "en":
            front_ends = k.accents
        else:
            front_ends = {spec.kokoro or f"espeak:{spec.espeak}": 1.0}
        engine = Kokoro(self.assets.kokoro)
        for s in todo:
            engine.generate(
                self.root / s.name,
                list(s.phrases),
                s.size,
                split_seed(self.args.seed, s.name),
                s.voices,
                front_ends,
                k.speed,
                s.blend,
            )
        del engine
        _free_gpu()

    def _piper(self, plan: list[Split], status: Callable[[str], None]) -> None:
        todo = [s for s in plan if count(self.root / s.name) < s.size]
        if todo:
            status("Piper")
        for s in todo:
            _piper.generate(
                self.assets.piper,
                self.root / s.name,
                list(s.phrases),
                s.size,
                split_seed(self.args.seed, s.name),
                self.args.piper,
            )
        _free_gpu()

    def _voxcpm(self, plan: list[Split], status: Callable[[str], None]) -> None:
        if all(count(self.root / s.name) >= s.size for s in plan):
            return
        status("VoxCPM2")
        v = self.args.voxcpm
        splits = [(s.name, s.voices, list(s.phrases), s.size) for s in plan]
        workers = [
            subprocess.Popen(
                _voxcpm.command(
                    self.root,
                    self.assets.voxcpm,
                    splits,
                    self.args.seed,
                    v.cfg_values,
                    v.inference_timesteps,
                    w,
                    v.workers,
                )
            )
            for w in range(v.workers)
        ]
        if any(worker.wait() for worker in workers):
            raise RuntimeError("a VoxCPM2 worker failed; rerun the same command to resume")


class _Monitor:
    """Report the number of clips on disk while engines write them."""

    def __init__(self, directories: list[Path], total: int, progress: Progress):
        self.directories, self.total, self.progress = directories, total, progress
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _poll(self) -> None:
        self.progress(sum(count(d) for d in self.directories), self.total)

    def _run(self) -> None:
        while not self._stop.wait(1.0):
            self._poll()

    def __enter__(self) -> _Monitor:
        self._poll()
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        self._poll()


def _free_gpu() -> None:
    import torch

    gc.collect()
    torch.cuda.empty_cache()
