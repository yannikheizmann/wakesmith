# Results

## Default arguments

An invented two-syllable wake word, trained with the default arguments and German negative
speech on an RTX 4090. Held out, through openWakeWord's runtime, at the recommended
threshold of 0.5:

| Measurement | Clean | Noisy |
|---|---:|---:|
| Recall, 606 clips from 6 unseen Kokoro voices | 94% | 85% |
| Recall, 217 clips from 10 unseen VoxCPM2 voice designs | 77% | 64% |

| Negative speech | Hours | False activations |
|---|---:|---:|
| openWakeWord validation speech (English) | 5.3 | 0 |
| MLS German test split | 5.6 | 0 |

## Trial run: "tomtom"

`uv run wakesmith "tomtom" --config trial.yaml` (Kokoro voices only, smaller datasets) finished
in 16 minutes on an RTX 4090:

| Measurement | Result |
|---|---:|
| Recall, 400 clips from 6 unseen Kokoro voices (clean / noisy) | 99.6% / 80.2% |
| False activations in 5.3 h of English speech | 0 |

## What changed the numbers

| Development step | Effect (validation unless noted) |
|---|---|
| Whisper screen instead of unscreened Piper positives | positives that actually say the word |
| Streaming features | removes a train/inference mismatch present in both upstream recipes |
| Hard-negative mining | large model: 0.56 → 0.19 false activations per hour at equal recall |
| Medium instead of large head | held out: 0 instead of 1.1 false activations per hour on English speech |
| Default-threshold-first selection | avoids thresholds supported by too little validation speech |
