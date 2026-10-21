---
title: Say "yambo" — wake-word demo
emoji: 🎙️
colorFrom: red
colorTo: yellow
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
short_description: Live demo of a wake-word model trained with wakesmith
---

# Say "yambo" — live wake-word demo

A wake-word classifier trained with
[wakesmith](https://github.com/yannikheizmann/wakesmith), running with the exact
openWakeWord streaming pipeline in the browser-free, native onnxruntime.

- Allow the microphone, press record, and say **"yambo"**.
- The meter shows the live model score; the detector fires on two consecutive
  frames above the recommended threshold of 0.5, with a two-second cooldown.
- Everything runs on the Space's CPU — no audio is stored.

Synthetic-data honest-notice: recognition quality depends on your microphone,
room and accent; the training data is synthetic. Train your own with
[wakesmith](https://github.com/yannikheizmann/wakesmith) for your voice.
