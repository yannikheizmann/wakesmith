# Training a wake word

```bash
uv run wakesmith "tomtom"
```

The command shows a live list of the six stages and a progress bar for the current one.
Library output is written to `logs/<stage>.log` in the run directory instead of the
terminal.

## How long it takes

The defaults (24,000 Kokoro and 5,000 VoxCPM2 positives, 72,000 training steps) take about
four hours on an RTX 4090 and produce the model you deploy. For a first try or to compare
spellings, use the [trial configuration](../configuration.md#a-faster-trial-run), which runs
in about 30 minutes.

## Common options

```bash
uv run wakesmith "Computer" --language de            # spoken in German
uv run wakesmith "nova" --negatives de               # English word in German households
uv run wakesmith "jarvis" -n "service" -n "nervous"  # extra phrases that must not activate
uv run wakesmith "tomtom" --no-voxcpm                # skip VoxCPM2 (less RAM, faster)
uv run wakesmith "tomtom" --config my-args.yaml      # any argument, see Configuration
```

## Spelling the word

The word is synthesised from its spelling. If the report shows few accepted positives, or
the clips in `clips/kokoro_positive_evaluation/` sound wrong, try a spelling closer to the
pronunciation you want ("tahmtahm" rather than "tomtom") and compare the reports.

## Resuming and reusing

Each run directory is named after the word and a hash of all arguments. Running the same
command again skips finished stages and resumes partially synthesised splits clip by clip.
Changing any setting creates a new run directory; downloads and negative-speech features are
shared between runs through the cache.

## Outputs

| File | Contents |
|---|---|
| `<word>.onnx` | the classifier |
| `<word>.json` | model card with the recommended threshold and held-out results |
| `report.md` | readable evaluation and data summary |
| `evaluation.json` | held-out measurements at all reported thresholds |
| `validation.json`, `training-log.json` | validation metrics used for selection |
| `clips/*/transcripts.json` | what Whisper heard for every synthetic clip |
