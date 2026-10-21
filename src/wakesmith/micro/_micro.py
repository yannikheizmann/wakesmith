"""microWakeWord backend: feature staging, training config, and TFLite export.

Wraps the `microwakeword` package (kahrendt/microWakeWord) so that a run with
``training.micro = true`` reuses wakesmith's synthesised positives and ambient
assets to train and export an int8 TFLite model with an ESPHome manifest.

The training and conversion run in a subprocess with the same interpreter; the
`micro` extra must be installed (`uv sync --extra micro`).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

NEGATIVE_SETS = [  # Precomputed micro-frontend spectrograms from kahrendt/microwakeword.
    ("negative_datasets/speech", 10.0, "random"),
    ("negative_datasets/dinner_party", 10.0, "random"),
    ("negative_datasets/no_speech", 5.0, "random"),
    ("negative_datasets/dinner_party_eval", 0.0, "split"),
]

MIXEDNET_DEFAULTS = {
    "pointwise_filters": "64,64,64,64",
    "repeat_in_block": "1, 1, 1, 1",
    "mixconv_kernel_sizes": "[5], [7,11], [9,15], [23]",
    "residual_connection": "0,0,0,0",
    "first_conv_filters": 32,
    "first_conv_kernel_size": 5,
    "stride": 3,
}


def stage_positive_spectrograms(clips_dir: Path, out_root: Path, assets, seed: int) -> dict:
    """Convert accepted wake-word clips into micro-frontend spectrogram mmaps.

    The accepted ``*_train``/``*_validation``/``*_evaluation`` split directories
    are each converted into one RaggedMmap feature folder with the micro speech
    frontend (40 features, 30 ms window, 10 ms step), augmented with wakesmith's
    impulse responses and background noise.
    """
    from microwakeword.audio.augmentation import Augmentation
    from microwakeword.audio.clips import Clips
    from microwakeword.audio.spectrograms import SpectrogramGeneration
    from mmap_ninja.ragged import RaggedMmap

    split_map = {"_train": "training", "_validation": "validation", "_evaluation": "testing"}
    rirs = sorted(assets.rirs.glob("**/*.wav"))
    backgrounds = sorted(assets.musan.glob("noise/**/*.wav")) + sorted(
        assets.musan.glob("music/**/*.wav")
    )
    augmenter = Augmentation(
        augmentation_probabilities={
            "SevenBandParametricEQ": 0.0,
            "TanhDistortion": 0.0,
            "PitchShift": 0.0,
            "BandStopFilter": 0.0,
            "AddColorNoise": 0.25,
            "AddBackgroundNoise": 0.75,
            "Gain": 1.0,
            "GainTransition": 0.25,
            "RIR": 0.5,
        },
        impulse_paths=[str(p) for p in rirs],
        background_paths=[str(p) for p in backgrounds],
    )
    counts = {}
    for split_dir in sorted(clips_dir.iterdir()):
        if not split_dir.is_dir() or "_positive" not in split_dir.name:
            continue
        target = next(v for k, v in split_map.items() if split_dir.name.endswith(k))
        wavs = sorted(split_dir.glob("*.wav"))
        if not wavs:
            continue
        specs = SpectrogramGeneration(
            clips=Clips(str(split_dir), "*.wav"),
            augmenter=augmenter if target == "training" else None,
        )
        out = out_root / "generated_augmented_features" / target
        out.mkdir(parents=True, exist_ok=True)
        RaggedMmap(out / f"positives_{seed}.mmap", generator=specs, batch_size=32)
        counts[target] = counts.get(target, 0) + 1
    return counts


def negative_features_root(run_dir: Path) -> Path:
    return run_dir / "micro" / "negative_datasets"


def download_negative_spectrograms(run_dir: Path) -> Path:
    """Fetch kahrendt's precomputed micro-frontend spectrograms for negatives."""
    from huggingface_hub import snapshot_download

    root = negative_features_root(run_dir)
    snapshot_download(
        "kahrendt/microwakeword",
        repo_type="dataset",
        local_dir=str(root.parent),
        allow_patterns=[
            "negative_datasets/speech/**",
            "negative_datasets/dinner_party/**",
            "negative_datasets/no_speech/**",
            "negative_datasets/dinner_party_eval/**",
        ],
    )
    return root


def write_training_config(args, run_dir: Path) -> Path:
    """Write the training_parameters.yaml that drives model_train_eval."""
    root = negative_features_root(run_dir)
    features = [
        {
            "features_dir": str(run_dir / "micro" / "generated_augmented_features" / split),
            "sampling_weight": 2.0 if split == "training" else 0.0 if split == "testing" else 1.0,
            "penalty_weight": 1.0,
            "truth": split == "training",
            "truncation_strategy": "truncate_start" if split == "training" else "split",
            "type": "mmap",
        }
        for split in ("training", "validation", "testing")
    ]
    for name, weight, trunc in NEGATIVE_SETS:
        if (root / name.replace("negative_datasets/", "")).exists():
            features.append(
                {
                    "features_dir": str(root / name.replace("negative_datasets/", "")),
                    "sampling_weight": weight,
                    "penalty_weight": 1.0,
                    "truth": False,
                    "truncation_strategy": trunc,
                    "type": "mmap",
                }
            )
    config = {
        "wake_word": args.wake_word,
        "window_step_ms": 10,
        "clip_duration_ms": 1500,
        "batch_size": 64,
        "training_steps": args.training.steps // 3,
        "evaluate_step_interval": 1000,
        "train_dir": str(
            run_dir / "micro" / "trained_models" / getattr(args, "slug", args.wake_word.lower())
        ),
        "features": features,
        "sample_rate": 16000,
        "micro_model": MIXEDNET_DEFAULTS,
    }
    path = run_dir / "micro" / "training_config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def train_micro_model(args, config_path: Path) -> tuple[Path, str]:
    """Run microwakeword's trainer/converter; return (tflite path, log text)."""
    config = yaml.safe_load(config_path.read_text())
    train_dir = Path(config["train_dir"])
    mixednet = config["micro_model"]
    command = [
        sys.executable,
        "-m",
        "microwakeword.model_train_eval",
        "--training_config",
        str(config_path),
        "--training_steps",
        str(config["training_steps"]),
        "--train",
        "1",
        "--restore_checkpoint",
        "1",
        "--test_tf_nonstreaming",
        "0",
        "--test_tflite_nonstreaming",
        "0",
        "--test_tflite_nonstreaming_quantized",
        "0",
        "--test_tflite_streaming",
        "0",
        "--test_tflite_streaming_quantized",
        "1",
        "--use_weights",
        "best_weights",
        "mixednet",
        *(a for pair in mixednet.items() for a in pair),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    tflite = next(
        (p for p in sorted(train_dir.rglob("*.tflite"), key=lambda p: "quantized" not in str(p))),
        None,
    )
    if tflite is None:
        raise FileNotFoundError("The microWakeWord training did not produce a .tflite model.")
    log = (result.stdout or "") + (result.stderr or "")
    return tflite, log


def estimate_tensor_arena(tflite_path: Path) -> int:
    """Best-effort tensor arena estimate in bytes for the ESPHome manifest."""
    try:
        import tensorflow as tf

        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        getter = getattr(interpreter, "_get_full_arena_size", None)
        if getter:
            return int(getter())
    except Exception:
        pass
    return 200_000


def build_manifest(args, tflite_path: Path, arena: int) -> dict:
    return {
        "type": "micro",
        "wake_word": args.wake_word,
        "author": "wakesmith",
        "website": "https://github.com/yannikheizmann/wakesmith",
        "model": tflite_path.name,
        "trained_languages": [args.language],
        "version": 2,
        "micro": {
            "probability_cutoff": args.training.default_threshold,
            "feature_step_size": 10,
            "sliding_window_size": 5,
            "tensor_arena_size": arena,
            "minimum_esphome_version": "2024.7.0",
        },
    }


def export(args, run_dir: Path, tflite: Path) -> tuple[Path, Path, dict]:
    """Copy the quantised model next to the run artefacts and write the manifest."""
    arena = estimate_tensor_arena(tflite)
    manifest = build_manifest(args, tflite, arena)
    out = run_dir / f"{args.slug}.tflite"
    out.write_bytes(tflite.read_bytes())
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return out, manifest_path, manifest
