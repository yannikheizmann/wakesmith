# Evaluation

The final report measures data that took no part in training or selection.

**Recall.** Clips from held-out Kokoro voices and VoxCPM2 descriptions are streamed through
openWakeWord's own `Model` after silence priming and counted with the activation gate
(threshold reached on two consecutive frames). Each clip is evaluated clean and with
augmentation from held-out noise files and rooms.

**False activations.** The held-out half of openWakeWord's validation speech and the MLS test
split (for other-language runs) are scored at every 80 ms position and counted with the same
gate, reported per hour of audio.

Results appear at the recommended threshold and at 0.3, 0.5, 0.7 and 0.9 in
`evaluation.json`, the model card and `report.md`.

!!! warning "What the evaluation cannot show"
    The positives are synthetic, and the negative speech is read or web speech rather than
    continuous audio from a room with a television. Measurements on a few hundred clips and a
    few hours of speech are samples, not guarantees. Record yourself and leave the model
    running in the target room before relying on it.
