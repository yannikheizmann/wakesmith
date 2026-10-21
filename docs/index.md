---
title: Custom wake words with one command
description: wakesmith trains openWakeWord-compatible wake-word models from synthetic speech, with runtime-identical features and honest held-out evaluation.
hide:
  - navigation
  - toc
---

[![Say "yambo" to a model trained with wakesmith — live in your browser](assets/wakesmith_banner.svg){ .figure }](#say-yambo-live-demo)

wakesmith trains a custom wake-word detector from nothing but the word itself. It
synthesises thousands of speakers saying it, keeps only the clips a speech recogniser hears
as the word, trains against 2,000 hours of other speech and reports how the model performs on
voices, rooms and speech it has never seen.

```bash
uv run wakesmith "tomtom"
```

<div class="homepage-actions" markdown>

[Try it live — say "yambo"](#say-yambo-live-demo){ .md-button .md-button--primary }
[Train your first wake word](guides/usage.md){ .md-button }
[How it works](method/index.md){ .md-button }

</div>

## Say "yambo" — live demo

A model trained with this pipeline runs right here. Click, allow the microphone,
say the word — everything stays in your browser.

<div id="wake-demo" class="wake-demo">
  <button id="wake-demo-start" type="button">Download model</button>
  <div class="wake-demo-progress"><div id="wake-demo-progress-bar"></div></div>
  <div class="wake-demo-meter">
    <div class="wake-demo-track"><div id="wake-demo-bar"></div><div class="wake-demo-threshold"></div></div>
    <span class="wake-demo-score-label">score <code id="wake-demo-score">0.000</code></span>
  </div>
  <div class="wake-demo-foot">
    <span>detections: <strong id="wake-demo-count">0</strong></span>
    <span id="wake-demo-status">click to download the model (about 2.5 MB, once per browser)</span>
  </div>
</div>

<script src="assets/wake-demo.js" defer></script>

## Runtime-identical features

Every clip is streamed through openWakeWord's own feature pipeline in 80 ms chunks, so the
classifier learns from exactly what it will see in deployment.

## Clips that say the word

Kokoro voice blends and VoxCPM2 voice designs speak the word; Whisper and a phonetic matcher
discard every clip that does not sound like it.

## Honest evaluation

False activations are counted per real hour with an integration's activation gate, the
threshold is chosen on validation data, and the report uses held-out voices, rooms and speech.

![The wakesmith pipeline](assets/wakesmith_pipeline.svg){ .figure }

!!! note "Scope"
    Synthetic voices and public speech corpora approximate real use. Test a trained model
    with your own microphone and room before relying on it.
