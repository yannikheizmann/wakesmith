<p align="center">
  <img src="docs/assets/wakesmith_banner.svg" alt="wakesmith: forge a custom wake word with one command" width="900">
</p>

<p align="center">
  <a href="https://yannikheizmann.github.io/wakesmith/"><img alt="Documentation" src="https://img.shields.io/badge/documentation-GitHub%20Pages-ff8a3d?style=flat-square"></a>
  <a href="https://github.com/yannikheizmann/wakesmith/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/yannikheizmann/wakesmith/ci.yml?branch=main&style=flat-square&label=CI"></a>
  <a href="pyproject.toml"><img alt="Python" src="https://img.shields.io/badge/python-3.11%20%7C%203.12-3776AB?style=flat-square&logo=python&logoColor=white"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/github/license/yannikheizmann/wakesmith?style=flat-square"></a>
  <img alt="openWakeWord compatible" src="https://img.shields.io/badge/output-openWakeWord%20ONNX-14172b?style=flat-square">
</p>

wakesmith trains a custom wake-word detector from nothing but the word itself. It
synthesises thousands of speakers saying it, checks every clip with a speech recogniser,
trains against 2,000 hours of other speech, and reports how the model does on voices,
rooms and speech it has never seen. The result is a small ONNX model that drops into any
[openWakeWord](https://github.com/dscripka/openWakeWord) integration.

```bash
uv run wakesmith "tomtom"
```

## Why another trainer?

Existing open-source recipes train on data their runtime never sees. wakesmith fixes the
issues measured while building a custom wake word:

- **Features identical to the runtime.** livekit-wakeword feeds `[-1, 1]` audio into
  openWakeWord's mel model, which expects int16-scale samples, so its embeddings differ
  completely from those used at inference time and from its own 2,000 h of negatives.
  openWakeWord's recipe extracts features in batch, which differs from streaming inference.
  wakesmith streams every clip through openWakeWord's own feature pipeline.
- **Clips that actually say the word.** Piper's LibriTTS voice, used by both recipes, rarely
  produces an intelligible invented word, and livekit-wakeword's wrapper deletes silent
  frames *inside* clips, cutting stop consonants and soft onsets. wakesmith synthesises mainly with Kokoro
  and VoxCPM2 and keeps a positive only if Whisper hears the word.
- **Honest numbers.** False activations are counted per real hour through the two-frame gate
  an integration uses, hard negatives are mined from all negative speech, the threshold is
  chosen on validation data only, and the final report uses held-out voices, rooms, noise
  and speech.

<p align="center">
  <img src="docs/assets/wakesmith_pipeline.svg" alt="Synthesise, screen, extract features, train and evaluate" width="900">
</p>

## Installation

wakesmith needs Linux on x86-64, an NVIDIA GPU with 16 GB or more (24 GB recommended for
the defaults), 32 GB of RAM and about 50 GB of disk. Everything else, including the
CUDA libraries and espeak-ng, comes from the locked environment:

```bash
git clone https://github.com/yannikheizmann/wakesmith.git
cd wakesmith
uv sync --frozen
```

## Usage

```bash
uv run wakesmith "tomtom"                          # ~4 h on an RTX 4090
uv run wakesmith "tomtom" --config trial.yaml      # smaller datasets, ~30 min
uv run wakesmith "Computer" --language de          # German pronunciation and negatives
uv run wakesmith "nova" --negatives de -n "noah"  # English word, German households
```

The first run downloads about 30 GB of pinned datasets and models into
`~/.cache/wakesmith`; later runs reuse them. Each run lives in
`wakesmith-runs/<word>-<recipe hash>/`, and running the same command again resumes after the
last finished stage. Every argument can be overridden from a YAML file; the
[configuration guide](https://yannikheizmann.github.io/wakesmith/configuration/) includes a
short trial configuration. The terminal shows a live stage list; library output goes to
`logs/<stage>.log`.

A finished run contains:

| File | Contents |
|---|---|
| `<word>.onnx` | openWakeWord-compatible classifier: `(batch, 16, 96)` embeddings to a score |
| `<word>.json` | model card: recommended threshold, held-out results and checksum |
| `report.md` | readable summary of the held-out evaluation and the synthetic data |
| `evaluation.json`, `training-log.json` | full measurements and validation history |

Load the model like any openWakeWord model and use the recommended threshold with two
consecutive frames:

```python
from openwakeword.model import Model

model = Model(
    wakeword_models=["wakesmith-runs/tomtom-<hash>/tomtom.onnx"], inference_framework="onnx"
)
score = model.predict(frame)["tomtom"]  # frame: 1280 int16 samples at 16 kHz
```

## Languages

`--language` selects how the word is pronounced, screened and contrasted: espeak-ng
phonemes, Whisper's language, Kokoro's front end, VoxCPM2's text, and confusable words from
the language's 60,000 most frequent words. Supported languages are English, German, Dutch,
French, Spanish, Italian, Portuguese and Polish. Non-English runs add speaker-disjoint
Multilingual LibriSpeech speech in the same language to the English negatives; `--negatives`
adds it to any run. The feature extractor itself is acoustic and language-agnostic.

## Results

An invented two-syllable wake word, trained with the default arguments and German negative
speech on an RTX 4090. Held out, at threshold 0.5:

| Measurement | Result |
|---|---:|
| Recall, unseen Kokoro voices (clean / noisy) | 94% / 85% |
| Recall, unseen VoxCPM2 voice designs (clean / noisy) | 77% / 64% |
| False activations in 5.3 h of English speech | 0 |
| False activations in 5.6 h of German speech | 0 |

The VoxCPM2 evaluation voices are deliberately hard: regional accents, elderly and
congested voices. A 16-minute trial run for "tomtom" (smaller datasets, Kokoro only) reached
99.6% / 80.2% clean / noisy recall with no false activations. Synthetic evaluations cannot
replace a test with your own microphone and room.

## Documentation

The [documentation](https://yannikheizmann.github.io/wakesmith/) explains each stage, the
recipe options, language support and the upstream issues in detail.

## Tests

```bash
uv sync --frozen --extra dev
uv run --frozen pytest -q
```

## Contributing and security

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the development workflow and conventions. Please
report vulnerabilities as described in [`SECURITY.md`](SECURITY.md).

## Citation

If wakesmith helps your work, please cite it using [`CITATION.cff`](CITATION.cff).

## License

wakesmith is available under the Apache License 2.0. Models and datasets it downloads keep
their own licences; see [`NOTICE`](NOTICE).
