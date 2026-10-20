"""Run the stages that turn a wake word into an evaluated ONNX model.

Each stage writes its outputs atomically and leaves a marker when it completes, so running
the same command again resumes after the last finished stage.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from wakesmith.config import (
    ARGS_FILENAME,
    EVALUATION_FILENAME,
    MODEL_CARD_FILENAME,
    MODEL_FILENAME,
    REPORT_FILENAME,
    TRAINING_LOG_FILENAME,
    Args,
)
from wakesmith.data import Assets, download, plan

from ._display import Display

STAGES = {
    "download": "Download data and models",
    "synthesise": "Synthesise speech",
    "screen": "Screen clips with Whisper",
    "features": "Extract runtime features",
    "train": "Train classifier",
    "evaluate": "Evaluate on held-out data",
}


class Pipeline:
    """One training run: its arguments, a shared cache and a run directory."""

    def __init__(self, args: Args, work_dir: Path, cache_dir: Path):
        self.args = args
        self.assets = Assets(cache_dir)
        self.run_dir = work_dir / f"{args.slug}-{args.digest}"
        self.clips, self.features = self.run_dir / "clips", self.run_dir / "features"
        self.model = self.run_dir / MODEL_FILENAME.format(slug=args.slug)

    def run(self) -> Path:
        """Execute all unfinished stages and return the path of the trained model."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(self.run_dir / ARGS_FILENAME, json.loads(self.args.model_dump_json()))
        os.environ.setdefault("HF_HOME", str(self.assets.root / "huggingface"))
        os.environ.setdefault("NLTK_DATA", str(self.assets.root / "nltk"))
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        subtitle = f"{self.args.language} · {self.run_dir}"
        with Display(
            f'wakesmith · "{self.args.wake_word}"', subtitle, STAGES, self.run_dir / "logs"
        ) as display:
            for key in STAGES:
                marker = self.run_dir / f".{key}.done"
                with display.stage(key, cached=marker.exists()):
                    if not marker.exists():
                        getattr(self, f"_{key}")(display)
                        marker.touch()
        return self.model

    def _download(self, display: Display) -> None:
        items = plan(self.assets, self.args)
        total, done = sum(size for _, _, size in items), 0
        display.detail("download", f"{len(items)} files" if items else "")

        def advance(n: int) -> None:
            nonlocal done
            done += n
            display.progress(done / 1e9, total / 1e9 or None, "gigabytes")

        for url, target, _ in items:
            download(url, target, advance)

    def _synthesiser(self):
        from wakesmith.synthesis import Synthesiser

        return Synthesiser(self.args, self.assets, self.clips)

    def _synthesise(self, display: Display) -> None:
        synthesiser = self._synthesiser()
        synthesiser.synthesise(
            lambda done, total: display.progress(done, total, "clips"),
            lambda engine: display.detail("synthesise", engine),
        )

    def _screen(self, display: Display) -> None:
        report = self._synthesiser().screen(
            lambda done, total: display.progress(done, total, "clips")
        )
        _write_json(self.clips / "screen.json", report)
        positives = sum(s["accepted"] for n, s in report.items() if "_positive_train" in n)
        display.detail("screen", f"{positives} training positives accepted")

    def _features(self, display: Display) -> None:
        from wakesmith.config import LANGUAGES
        from wakesmith.features import clip_features, noise_sources, speech_features

        r, synthesiser = self.args, self._synthesiser()
        sources = noise_sources(self.assets.musan, self.assets.rirs)
        jobs = []
        for index, split in enumerate(
            s for s in synthesiser.splits() if "_evaluation" not in s.name
        ):
            files = synthesiser.accepted(split.name)
            rounds = (
                r.augmentation.rounds
                if split.name.endswith("_train")
                else r.augmentation.validation_rounds
            )
            if files:
                jobs.append((split.name, files, rounds, r.seed + 1000 * index))
        total = sum(len(f) * n for _, f, n, _ in jobs) + r.augmentation.background_clips
        done = 0

        def advance(n: int) -> None:
            nonlocal done
            done += n
            display.progress(done, total, "clips")

        for name, files, rounds, seed in jobs:
            clip_features(files, self.features / f"{name}.npy", sources, rounds, seed, advance)
        clip_features(
            [None] * r.augmentation.background_clips,
            self.features / "background_train.npy",
            sources,
            1,
            r.seed + 77,
            advance,
            background=True,
        )
        validation = np.load(self.assets.validation, mmap_mode="r")
        half = (
            len(validation) // 2
        )  # First half for selection; second half, after a gap, for evaluation.
        for name, part in (
            ("owwval_validation", validation[:half]),
            ("owwval_evaluation", validation[half + 16 :]),
        ):
            if not (self.features / f"{name}.npy").exists():
                np.save(self.features / f"{name}.npy", np.asarray(part, dtype=np.float16))
        speech = r.negative_speech
        if speech.language:
            shards = self.assets.mls_shards(LANGUAGES[speech.language].mls, speech.train_shards)
            hours = {
                "train": speech.train_hours,
                "validation": speech.validation_hours,
                "evaluation": speech.evaluation_hours,
            }
            done, total = 0.0, sum(hours.values())

            def advance_hours(h: float) -> None:
                nonlocal done
                done += h
                display.progress(round(done, 1), total, "hours of speech")

            for split, h in hours.items():
                # Word-independent, so cached for every later run with the same settings.
                key = hashlib.sha256(
                    f"{speech.language}-{split}-{h}-{speech.train_shards}-{r.seed}".encode()
                )
                shared = self.assets.root / "speech-features" / f"{key.hexdigest()[:12]}.npz"
                speech_features(
                    shards[split],
                    shared,
                    h,
                    sources,
                    r.seed + 5000,
                    split == "train",
                    advance_hours,
                )
                link = self.features / f"speech_{split}.npz"
                if not link.exists():
                    link.symlink_to(shared)

    def _train(self, display: Display) -> None:
        import torch

        from wakesmith.training import TrainingData, export_onnx, train

        data = TrainingData(self.features, self.assets.acav, self.args, torch.device("cuda"))
        result = train(
            self.args, data, lambda done, total, label: display.progress(done, total, label)
        )
        display.detail("train", f"recommended threshold {result.threshold}")
        _write_json(self.run_dir / TRAINING_LOG_FILENAME, result.log)
        _write_json(
            self.run_dir / "validation.json",
            {"threshold": result.threshold, "metrics": result.metrics},
        )
        export_onnx(result.model, self.model)

    def _evaluate(self, display: Display) -> None:
        from wakesmith.evaluation import false_activations, model_card, recall, report
        from wakesmith.features import noise_sources

        r, synthesiser = self.args, self._synthesiser()
        validation = json.loads((self.run_dir / "validation.json").read_text())
        threshold = validation["threshold"] or r.training.default_threshold
        thresholds = sorted({*r.evaluation.thresholds, threshold})
        sources = noise_sources(self.assets.musan, self.assets.rirs, evaluation=True)
        sets = {
            name: synthesiser.accepted(name)
            for name in ("kokoro_positive_evaluation", "voxcpm_positive_evaluation")
            if (self.clips / name).exists()
        }
        steps, done = 2 * len(sets) + 2, 0
        recalls = {}
        for name, files in sets.items():
            for condition in ("clean", "noisy"):
                label = f"{name.split('_')[0]} unseen voices, {condition}"
                display.progress(done, steps, label)
                recalls[f"{name.split('_')[0]}_unseen_voices/{condition}"] = recall(
                    self.model,
                    files,
                    sources,
                    condition == "noisy",
                    r.seed + 1,
                    thresholds,
                    r.evaluation,
                )
                done += 1
        streams = {
            "openwakeword_validation_heldout": (
                np.load(self.features / "owwval_evaluation.npy"),
                None,
            )
        }
        if (self.features / "speech_evaluation.npz").exists():
            speech = np.load(self.features / "speech_evaluation.npz")
            streams[f"{r.negative_speech.language}_speech_heldout"] = (
                speech["frames"],
                speech["offsets"],
            )
        negative = {}
        for name, (frames, offsets) in streams.items():
            display.progress(done, steps, name.replace("_", " "))
            offsets = offsets if offsets is not None else np.array([0, len(frames)])
            hours, rates = false_activations(self.model, frames, offsets, thresholds, r.evaluation)
            negative[name] = {"hours": hours, "false_activations_per_hour": rates}
            done += 1
        evaluation = {
            "thresholds": [str(t) for t in thresholds],
            "recall": recalls,
            "negative_streams": negative,
        }
        _write_json(self.run_dir / EVALUATION_FILENAME, evaluation)
        sha = hashlib.sha256(self.model.read_bytes()).hexdigest()
        card = model_card(r, sha, threshold, evaluation, validation["metrics"])
        _write_json(self.run_dir / MODEL_CARD_FILENAME.format(slug=r.slug), card)
        report(
            card,
            evaluation,
            json.loads((self.clips / "screen.json").read_text()),
            self.run_dir / REPORT_FILENAME,
        )


def _write_json(path: Path, value) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=2) + "\n")
    partial.replace(path)
