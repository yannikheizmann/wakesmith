---
title: Custom wake words with one command
description: wakesmith trains openWakeWord-compatible wake-word models from synthetic speech, with runtime-identical features and honest held-out evaluation.
hide:
  - navigation
  - toc
---

![wakesmith: forge a custom wake word with one command](assets/wakesmith_banner.svg){ .figure }

wakesmith trains a custom wake-word detector from nothing but the word itself. It
synthesises thousands of speakers saying it, keeps only the clips a speech recogniser hears
as the word, trains against 2,000 hours of other speech and reports how the model performs on
voices, rooms and speech it has never seen.

```bash
uv run wakesmith "tomtom"
```

<div class="homepage-actions" markdown>

[Try "yambo" — live demo on Hugging Face](https://huggingface.co/spaces/yannikheizmann/wakesmith-demo){ .md-button }
[Train your first wake word](guides/usage.md){ .md-button .md-button--primary }
[How it works](method/index.md){ .md-button }

</div>

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
