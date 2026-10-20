# Upstream findings

These issues were measured while building a custom wake word. Each affects models trained
with the respective upstream recipe.

## livekit-wakeword feeds the wrong sample scale to the mel model

openWakeWord's `melspectrogram.onnx` expects int16-scale samples (±32,768); openWakeWord and its
published ACAV100M features use that scale. livekit-wakeword's `MelSpectrogramFrontend` passes
`[-1, 1]` floats. On the same speech, the resulting embeddings differ by up to 28 against a
mean magnitude of 13, while int16 input reproduces openWakeWord to within 5 × 10⁻⁵. As a
result, livekit-wakeword trains positives at one scale against negatives (the shared ACAV100M
features) at another, and its models receive differently scaled features in openWakeWord
runtimes.

## Batch features differ from streaming inference

openWakeWord's recipe embeds each clip in one batch call. Streaming inference computes the mel
spectrogram chunk by chunk with a short overlap and aligns embedding windows to the newest
chunk. On speech the two agree with a correlation of about 0.97 (mean absolute difference ≈ 3
against magnitudes ≈ 13).

## Piper clips lose their consonants

livekit-wakeword's Piper wrapper keeps only frames that WebRTC VAD classifies as speech,
anywhere in the clip, and concatenates them. Stop closures and soft onsets are removed, so an
invented word starting with a /j/ glide was transcribed without its onset ("ambo", "embo").
Even with only the edges trimmed, Whisper recognised fewer than one in ten isolated clips of
that word from the LibriTTS voice, against 26 of 28 for Kokoro.

## False activations estimated per window, not per streamed hour

livekit-wakeword estimates false activations per hour from non-overlapping 2 s windows of
validation speech and a 2 s duration per window, although each window covers 1.28 s, and a
deployed model is scored every 80 ms. wakesmith scores every position and counts activations
with an integration's two-frame gate.
