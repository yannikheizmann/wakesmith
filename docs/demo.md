# Try a trained model in your browser

This page runs a real wake-word model — trained with this pipeline for the word
**“yambo”** — entirely in your browser. Nothing is recorded or uploaded: the model
and your microphone audio never leave the page.

Press start, allow the microphone, and say the word. The bar shows the live model
score; the detector fires on two consecutive frames above the recommended
threshold of 0.5, with a two-second cooldown — the same gate an integration uses.

<div id="wake-demo" class="wake-demo">
  <div class="wake-demo-head">
    <button id="wake-demo-button" type="button">Start listening</button>
    <span id="wake-demo-status">Click start to begin. The model loads once (about 3 MB).</span>
  </div>
  <div class="wake-demo-meter">
    <div class="wake-demo-track"><div id="wake-demo-bar"></div><div class="wake-demo-threshold"></div></div>
    <span class="wake-demo-score-label">score <code id="wake-demo-score">0.000</code></span>
  </div>
  <div class="wake-demo-foot">
    <span>detections: <strong id="wake-demo-count">0</strong></span>
    <span id="wake-demo-gate">gate: 0/2 frames</span>
  </div>
</div>

The demo runs the classifier and embedding models with onnxruntime-web
(WebAssembly) and computes the mel front end natively from the model's own
DFT weights — streaming 80 ms chunks through the same pipeline the training
and evaluation use, including the silence priming and the exact buffer
semantics. What you see here is what an integration gets.

Perfect recognition is not the point: your microphone, room and browser
resampling differ from the synthetic training data. If it struggles, that is the
honesty this tool is built on — [train your own](guides/usage.md) for your voice.

<style>
  .wake-demo {
    background: #2a0000; border-radius: 12px; color: #ffeaa5;
    font-family: var(--md-text-font-family, Inter, sans-serif);
    margin: 1.5rem 0; padding: 1.25rem 1.5rem;
  }
  .wake-demo.detected { box-shadow: 0 0 0 3px #da8c8c; }
  .wake-demo-head { align-items: center; display: flex; gap: 1rem; margin-bottom: 1rem; }
  .wake-demo button {
    background: #ffeaa5; border: none; border-radius: 8px; color: #2a0000;
    cursor: pointer; font-size: 0.9rem; font-weight: 700; padding: 0.6rem 1.2rem;
  }
  .wake-demo button.listening { background: #da8c8c; color: #2a0000; }
  .wake-demo button:disabled { opacity: 0.6; cursor: wait; }
  .wake-demo-meter { align-items: center; display: flex; gap: 1rem; margin: 0.75rem 0; }
  .wake-demo-track {
    background: #3a0a0a; border-radius: 6px; flex: 1; height: 14px; overflow: hidden; position: relative;
  }
  .wake-demo-track > div {
    background: #da8c8c; height: 100%; left: 0; position: absolute; top: 0;
    transition: width 80ms linear; width: 0;
  }
  .wake-demo-threshold { border-left: 2px solid #ffeaa5; height: 100%; left: 50%; opacity: 0.5; position: absolute; top: 0; }
  .wake-demo-score-label { color: #ffeaa5; font-size: 0.85rem; white-space: nowrap; }
  .wake-demo-foot { color: #da8c8c; display: flex; font-size: 0.8rem; gap: 1.5rem; justify-content: space-between; }
  .wake-demo-foot strong { color: #ffeaa5; }
</style>

<script src="../assets/wake-demo.js" defer></script>
