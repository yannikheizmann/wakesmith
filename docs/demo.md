---
title: Say "yambo"
hide:
  - navigation
  - toc
---

![wakesmith: forge a custom wake word with one command](assets/wakesmith_banner.svg){ .figure }

<div class="wake-demo-say">Say <span class="wake-demo-word">“yambo”</span></div>

<div id="wake-demo" class="wake-demo">
  <button id="wake-demo-start" type="button">Download model</button>
  <div class="wake-demo-meter">
    <div class="wake-demo-track"><div id="wake-demo-bar"></div><div class="wake-demo-threshold"></div></div>
    <span class="wake-demo-score-label">score <code id="wake-demo-score">0.000</code></span>
  </div>
  <div class="wake-demo-foot">
    <span>detections: <strong id="wake-demo-count">0</strong></span>
    <span id="wake-demo-status">click to download the model (about 2.5 MB, once per browser)</span>
  </div>
</div>

Everything runs in your browser; nothing is recorded or uploaded. The model was
trained with this pipeline — [train your own](guides/usage.md).

<script src="../assets/wake-demo.js" defer></script>
