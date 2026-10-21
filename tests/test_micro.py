"""Unit tests for the microWakeWord manifest and training config builders."""

from pathlib import Path
from types import SimpleNamespace

import yaml

from wakesmith.micro import build_manifest, write_training_config


def _args(steps=6000, threshold=0.5, word="tomtom", language="en"):
    return SimpleNamespace(
        wake_word=word,
        language=language,
        training=SimpleNamespace(steps=steps, default_threshold=threshold),
    )


def test_manifest_follows_the_esphome_schema():
    manifest = build_manifest(_args(), Path("yambo.tflite"), 22860)
    assert manifest["type"] == "micro"
    assert manifest["model"] == "yambo.tflite"
    assert manifest["trained_languages"] == ["en"]
    micro = manifest["micro"]
    assert micro["probability_cutoff"] == 0.5
    assert micro["feature_step_size"] == 10
    assert micro["sliding_window_size"] == 5
    assert micro["tensor_arena_size"] == 22860
    assert micro["minimum_esphome_version"] == "2024.7.0"


def test_training_config_lists_positive_and_negative_sets(tmp_path: Path):
    run_dir = tmp_path / "run"
    neg = run_dir / "micro" / "negative_datasets"
    for name in ("speech", "dinner_party", "no_speech", "dinner_party_eval"):
        (neg / name).mkdir(parents=True)
    config_path = write_training_config(_args(), run_dir)
    config = yaml.safe_load(config_path.read_text())
    dirs = [f["features_dir"] for f in config["features"]]
    assert any("generated_augmented_features/training" in d for d in dirs)
    assert sum("negative_datasets/speech" in d for d in dirs) == 1
    assert all(Path(d).exists() for d in dirs if "negative_datasets" in d)
    assert config["training_steps"] == 2000
    assert config["micro_model"]["stride"] == 3
