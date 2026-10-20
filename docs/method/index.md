# Method overview

![The wakesmith pipeline](../assets/wakesmith_pipeline.svg){ .figure }

wakesmith turns a word into a model in six resumable stages:

1. **Download** pinned datasets and weights into the shared cache.
2. **Synthesise** positives and confusable negatives with three text-to-speech engines
   ([details](synthesis.md)).
3. **Screen** every positive with Whisper and a phonetic matcher, discarding clips that do not
   sound like the wake word.
4. **Extract features** after acoustic augmentation, streaming each clip through openWakeWord's
   own pipeline ([details](features.md)).
5. **Train** livekit-wakeword's conv-attention classifier with streaming validation and
   hard-negative mining, and choose the threshold on validation data ([details](training.md)).
6. **Evaluate** on held-out voices, noise, rooms and speech through openWakeWord's runtime and
   an activation gate ([details](evaluation.md)).

## Data splits

| Data | Training | Validation (selection) | Evaluation (report) |
|---|---|---|---|
| Kokoro speakers | blends of 48 voices | blends of 48 voices | 6 voices never blended |
| VoxCPM2 speakers | 28 voice descriptions | same descriptions | 10 other descriptions |
| Noise and rooms | 90% of MUSAN and MIT RIRs | same | held-out 10% by file hash |
| English speech | 95% of ACAV100M (2,000 h) | last 5% of ACAV100M and half of openWakeWord's validation set | other half of the validation set |
| Other-language speech | MLS train shards | MLS dev | MLS test (speaker-disjoint) |

Nothing in the evaluation column influences training, checkpoint averaging or the
recommended threshold.
