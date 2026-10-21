---
title: Say "yambo" — wake-word demo
emoji: 🎙️
colorFrom: red
colorTo: yellow
sdk: static
pinned: false
license: apache-2.0
short_description: Say "yambo" to a model trained with wakesmith
---

A live demo of the wake-word model trained with
[wakesmith](https://github.com/yannikheizmann/wakesmith). Everything runs in
your browser: the mel front end is computed from the model's own DFT weights
and the classifier runs with onnxruntime-web. The streaming pipeline was
verified numerically against openWakeWord's runtime (frame-by-frame).
