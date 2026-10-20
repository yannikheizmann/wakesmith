# Runtime-identical features

openWakeWord models do not see audio: a fixed mel-spectrogram model and Google's speech
embedding model turn every 80 ms of 16 kHz audio into one 96-dimensional frame, and the
classifier scores the last 16 frames.

## Streaming, not batch

wakesmith feeds each training clip to openWakeWord's `AudioFeatures` in 1280-sample chunks,
starting from the state the runtime reaches after priming with 3.2 s of silence, and records
the frame produced after every chunk. A random 0–1279-sample offset moves the chunk
boundaries relative to the speech, as a live microphone does. The resulting windows are
identical to those of a streaming runtime to within 10⁻⁴; a test enforces this.

openWakeWord's training recipe instead embeds whole clips in batch, which differs from
streaming at chunk edges (correlation ≈ 0.97 on speech). livekit-wakeword passes `[-1, 1]`
floats to the mel model, which expects int16-scale samples, producing embeddings that
differ by up to 28 against a typical magnitude of 13.

## Augmentation

Each training clip is placed at the end of a 2 s window with 0–200 ms of trailing context and
augmented three times with openWakeWord's chain plus wider level
and noise coverage:

- seven-band equalisation, mild distortion, ±3 semitone pitch shift and band-stop filtering;
- room impulse responses (50%);
- MUSAN music, noise or babble at −5 to 20 dB signal-to-noise ratio (75%);
- coloured noise (25%);
- a peak level between −36 and −1 dBFS.

Pure background windows serve as standalone negatives. Continuous negative speech is
augmented with rooms and noise on half of its recordings.
