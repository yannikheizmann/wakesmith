# Configuration

Every run is defined by one typed `Args` object composed of argument groups. Values resolve in this order:

1. field defaults, the full recipe for a deployable model;
2. a YAML file passed with `--config`;
3. command-line options.

Unknown fields are rejected. The run directory is named after a hash of all arguments,
so changing any value starts a new run.

## A faster trial run

The defaults train the model you deploy (~4 h on an RTX 4090). To try a word or compare
spellings in about 30 minutes, override the dataset sizes:

```yaml
# trial.yaml
kokoro: {n_train: 6000, n_validation: 600, n_evaluation: 400,
         n_adversarial_train: 4000, n_adversarial_validation: 400}
voxcpm: {enabled: false}
piper: {n_adversarial_train: 8000, n_adversarial_validation: 800}
augmentation: {rounds: 2, background_clips: 2000}
training: {steps: 20000, hard_negatives: 10000}
```

```bash
uv run wakesmith "tomtom" --config trial.yaml
```

## Example

```yaml
# my-args.yaml
kokoro:
  n_train: 30000
  speed: [0.8, 1.2]
voxcpm:
  enabled: true
  workers: 1
negative_speech:
  language: de
  train_hours: 100
training:
  steps: 80000
  target_false_activations_per_hour: 0.1
evaluation:
  consecutive_frames: 3
```

## Argument reference

| Group | Field | Default | Meaning |
|---|---|---|---|
| — | `language` | `en` | spoken language |
| — | `extra_negative_phrases` | `[]` | phrases that must not activate |
| `kokoro` | `n_train`, `n_validation`, `n_evaluation` | 24,000 / 1,500 / 800 | positive clips |
| `kokoro` | `n_adversarial_train`, `n_adversarial_validation` | 12,000 / 1,000 | confusable clips |
| `kokoro` | `accents` | US 45%, UK 25%, others 6% each | English front ends |
| `kokoro` | `speed` | 0.75–1.25 | speaking rate |
| `voxcpm` | `enabled`, `workers` | true, 1 | VoxCPM2 synthesis and processes |
| `voxcpm` | `n_train` … `n_adversarial_validation` | 5,000 … 200 | clip counts |
| `piper` | `n_adversarial_train`, `n_adversarial_validation` | 30,000 / 2,000 | confusable clips |
| `screen` | `model`, `revision` | Whisper large-v3-turbo | screening model |
| `screen` | `vowel_tolerance` | 0.5 | allowed vowel difference |
| `negative_speech` | `language` | none for English | MLS language |
| `negative_speech` | `train_hours`, `validation_hours`, `evaluation_hours` | 60 / 6 / 6 | hours of speech |
| `augmentation` | `rounds`, `background_clips` | 3, 4,000 | augmented copies, noise windows |
| `training` | `model_size` | medium | tiny, small, medium or large |
| `training` | `micro` | false | train a microWakeWord int8 TFLite model (ESP32/Voice PE) instead of openWakeWord ONNX; needs `uv sync --extra micro` |
| `training` | `steps` | 60,000 | first-phase steps |
| `training` | `hard_negatives` | 20,000 | mined windows (0 disables) |
| `training` | `target_false_activations_per_hour` | 0.2 | validation limit |
| `training` | `max_adversarial_accept` | 0.01 | validation limit |
| `training` | `default_threshold` | 0.5 | preferred threshold |
| `training.batch` | `kokoro`, `voxcpm`, `adversarial`, `acav`, `speech`, `background`, `hard` | 80, 48, 128, 1024, 256, 64, 128 | examples per step |
| `evaluation` | `consecutive_frames`, `cooldown_ms` | 2, 2000 | activation gate |
