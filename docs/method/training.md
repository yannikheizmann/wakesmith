# Training

## Classifier

The head is livekit-wakeword's temporal conv-attention classifier (`medium`: 128 channels, two
convolution blocks, multi-head self-attention, mean pooling and a sigmoid output). Unlike a
flattening DNN head it keeps the order of the 16 frames.

## Batches

Every step mixes, by default, 80 Kokoro and 48 VoxCPM2 positives with 128 confusable clips,
1,024 windows of ACAV100M speech, 64 background windows, 256 windows of other-language speech
when configured and, after mining, 128 hard negatives. ACAV100M is read from disk in random
64-row blocks on a background thread; everything else lives on the GPU.

## Optimisation

Focal loss (γ = 2), embedding mixup (α = 0.2) and label smoothing (0.05) follow
livekit-wakeword. Negative examples are weighted up linearly to 1,500 within each phase. Three
phases run at decreasing learning rates (100%, 10%, 1% of 10⁻⁴; the second and third are a tenth
as long as the first). If validation still shows too many false activations before a phase,
the maximum negative weight doubles.

## Hard-negative mining

After the first phase the model scores all 2,000 hours of ACAV100M and any other-language
speech. The 20,000 highest-scoring windows, the moments of speech that most resemble the
wake word, are oversampled for the remaining phases. In development this reduced validation
false activations of the large model from 0.56 to 0.19 per hour at an unchanged recall.

## Validation and the recommended threshold

During the last quarter of each phase the model is validated twenty times on:

- positives and confusables from the validation splits;
- openWakeWord's validation speech and any other-language speech, scored at every 80 ms
  position and counted with the two-frame, two-second activation gate;
- ~100 hours of held-out ACAV100M windows.

The five best checkpoints are averaged. The recommended threshold is 0.5 if it meets
≤ 0.2 false activations per hour on every stream and ≤ 1% accepted confusables, otherwise the
lowest higher threshold that does. Lower thresholds are never proposed: one activation in a
few hours of validation speech is too little evidence to trade false activations for recall.
