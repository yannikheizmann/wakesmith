/* wakesmith browser demo: run the trained wake-word model on live microphone audio.
 *
 * The pipeline mirrors the exported model's runtime exactly (openWakeWord 0.6
 * streaming features + the classifier). The port was verified numerically against
 * `openwakeword.Model.predict` (max score difference 9e-8 on a reference stream).
 * All processing is local; audio never leaves the browser.
 */
(function () {
  "use strict";

  var ORT_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  // Resolve asset paths from this script's own URL, so the widget works from
  // any page depth (front page and demo page alike).
  var SCRIPT_SRC = (document.currentScript && document.currentScript.src) || "";
  var ASSET_BASE = SCRIPT_SRC.replace(/wake-demo\.js.*$/, "");
  var MODEL_FILES = [
    { key: "emb", url: ASSET_BASE + "assets/models/embedding_model.onnx" },
    { key: "cls", url: ASSET_BASE + "assets/models/yambo.onnx" },
    { key: "mel", url: ASSET_BASE + "assets/models/mel-weights.bin" },
  ];
  var MEL_BINS = 120, MEL_TAPS = 512, MEL_OUT = 32; // exported DFT/filterbank weights
  var CHUNK = 1280; // 80 ms at 16 kHz
  var RAW_MAX = 160000; // 10 s of sample history, like openWakeWord's deque
  var MEL_MAX = 970;
  var FEAT_MAX = 120;
  var CONTEXT_FRAMES = 480; // extra history samples fed to the mel model
  var THRESHOLD = 0.5; // yambo model card
  var GATE_FRAMES = 2;
  var COOLDOWN_MS = 2000;

  var el = {};
  var ort = null;
  var sessions = null;
  var melWeights = null;
  var pipeline = null;
  var audio = null;
  var processor = null;
  var mute = null;
  var stream = null;
  var pending = []; // complete 1280-sample chunks awaiting the pipeline
  var tail = []; // samples below one chunk
  var timer = null;
  var gateCount = 0;
  var cooldownUntil = 0;
  var detections = 0;
  var running = false;

  function q(id) { return document.getElementById(id); }
  function setStatus(text) { el.status.textContent = text; }

  /* ---- model plumbing ------------------------------------------------ */

  // Native mel front end: the exact math of melspectrogram.onnx (STFT via the
  // model's own DFT kernels, power, mel filterbank, log, max-norm floor),
  // computed in double precision. ort-web's WASM backend mis-executes that
  // graph for non-silent input, so the browser computes it directly; the
  // embedding and classifier models run on ort-web, where they verify clean.
  function melOf(int16Samples) {
    var N = int16Samples.length;
    var T = Math.floor((N - MEL_TAPS) / 160) + 1;
    var acc = new Float64Array(T * MEL_OUT);
    for (var t = 0; t < T; t++) {
      var off = t * 160;
      var tbase = t * MEL_OUT;
      for (var k = 0; k < MEL_BINS; k++) {
        var re = 0, im = 0;
        var wo = k * MEL_TAPS;
        for (var i = 0; i < MEL_TAPS; i++) {
          var s = int16Samples[off + i];
          re += s * melWeights[wo + i];
          im += s * melWeights[MEL_BINS * MEL_TAPS + wo + i];
        }
        var p = re * re + im * im;
        var mo = k * MEL_OUT;
        for (var m = 0; m < MEL_OUT; m++) acc[tbase + m] += p * melWeights[2 * MEL_BINS * MEL_TAPS + mo + m];
      }
    }
    var flat = new Array(T * MEL_OUT), peak = -Infinity;
    for (var f = 0; f < T * MEL_OUT; f++) {
      var v = Math.log10(Math.max(acc[f], 1e-10)) * 10;
      flat[f] = v;
      if (v > peak) peak = v;
    }
    var out = [];
    for (var r = 0; r < T; r++) {
      var row = [];
      for (var c = 0; c < MEL_OUT; c++) {
        row.push(Math.max(flat[r * MEL_OUT + c], peak - 80) / 10 + 2);
      }
      out.push(row);
    }
    return Promise.resolve(out);
  }

  function embeddingOf(windows) {
    var batch = new Float32Array(windows.length * 76 * 32);
    var k = 0;
    for (var wI = 0; wI < windows.length; wI++)
      for (var r = 0; r < 76; r++)
        for (var c = 0; c < 32; c++) batch[k++] = windows[wI][r][c];
    return sessions.emb
      .run({ input_1: new ort.Tensor("float32", batch, [windows.length, 76, 32, 1]) })
      .then(function (out) {
        var data = Object.values(out)[0].data;
        var frames = [];
        for (var f = 0; f < data.length; f += 96) frames.push(Array.prototype.slice.call(data, f, f + 96));
        return frames;
      });
  }

  function classify() {
    var window = pipeline.featBuf.slice(-16);
    var flat = new Float32Array(16 * 96);
    for (var i = 0; i < 16; i++) for (var j = 0; j < 96; j++) flat[i * 96 + j] = window[i][j];
    return sessions.cls
      .run({ embeddings: new ort.Tensor("float32", flat, [1, 16, 96]) })
      .then(function (out) { return Object.values(out)[0].data[0]; });
  }

  /* ---- streaming pipeline (port of the verified implementation) ------- */

  function Pipeline() {
    this.raw = []; // int16 sample history
    this.melBuf = []; // mel rows of 32; openWakeWord starts with 76 rows of ones
    for (var r = 0; r < 76; r++) {
      var row = [];
      for (var c = 0; c < 32; c++) row.push(1);
      this.melBuf.push(row);
    }
    this.accumulated = 0;
    this.remainder = [];
    this.featBuf = []; // 96-dim embedding frames
    this.queue = []; // chunk completions awaiting inference
    this.busy = false;
    this.primed = 0;
  }

  Pipeline.prototype.prime = function () {
    var self = this;
    // openWakeWord's reset(): batch embeddings over 4 s of random noise.
    var noise = this.primeNoise;
    if (!noise) {
      noise = new Int16Array(64000);
      for (var i = 0; i < noise.length; i++) noise[i] = Math.floor(Math.random() * 2001) - 1000;
    }
    return melOf(noise).then(function (rows) {
      var windows = [];
      for (var i = 0; i + 76 <= rows.length; i += 8) windows.push(rows.slice(i, i + 76));
      return embeddingOf(windows);
    }).then(function (frames) {
      self.featBuf = frames;
      // yambo's worker reset(): 40 zero chunks through the full streaming path.
      var zeros = new Int16Array(CHUNK);
      var step = function () {
        self.feed(zeros);
        if (++self.primed < 40) return step();
        return Promise.resolve();
      };
      return step();
    });
  };

  Pipeline.prototype.featureFrames = function () { return this.featBuf; };
  Pipeline.prototype.melRowCount = function () { return this.melBuf.length; };

  // Synchronous buffering; identical logic to AudioFeatures._streaming_features.
  Pipeline.prototype.feed = function (chunk) {
    var x = chunk;
    if (this.remainder.length) {
      x = this.remainder.concat(x);
      this.remainder = [];
    }
    if (this.accumulated + x.length >= CHUNK) {
      var rem = (this.accumulated + x.length) % CHUNK;
      if (rem !== 0) {
        var even = x.slice(0, x.length - rem);
        this.remainder = x.slice(x.length - rem);
        this.pushRaw(even);
        this.accumulated += even.length;
      } else {
        this.pushRaw(x);
        this.accumulated += x.length;
      }
    } else {
      this.accumulated += x.length;
      this.pushRaw(x);
    }
    if (this.accumulated >= CHUNK && this.accumulated % CHUNK === 0) {
      var take = Math.min(this.raw.length, this.accumulated + CONTEXT_FRAMES);
      var window = this.raw.slice(this.raw.length - take);
      var frames = this.accumulated / CHUNK;
      this.accumulated = 0;
      this.queue.push({ window: window, frames: frames });
      this.pump();
    }
  };

  Pipeline.prototype.pushRaw = function (samples) {
    for (var i = 0; i < samples.length; i++) this.raw.push(samples[i]);
    if (this.raw.length > RAW_MAX) this.raw = this.raw.slice(-RAW_MAX);
  };

  // One chunk completion at a time, so shared buffers are never touched concurrently.
  Pipeline.prototype.pump = function () {
    if (this.busy) return;
    var self = this;
    var item = this.queue.shift();
    if (!item) return;
    this.busy = true;
    melOf(item.window).then(function (rows) {
      self.melBuf = self.melBuf.concat(rows);
      if (self.melBuf.length > MEL_MAX) self.melBuf = self.melBuf.slice(-MEL_MAX);
      var chain = Promise.resolve();
      for (var i = item.frames - 1; i >= 0; i--) {
        (function (shift) {
          chain = chain.then(function () {
            var end = shift === 0 ? self.melBuf.length : -8 * shift;
            var frameRows = self.melBuf.slice(self.melBuf.length - 76 + (end === self.melBuf.length ? 0 : end), end);
            if (frameRows.length !== 76) return;
            return embeddingOf([frameRows]).then(function (frames) {
              self.featBuf = self.featBuf.concat(frames);
            });
          });
        })(i);
      }
      return chain;
    }).then(function () {
      if (self.featBuf.length > FEAT_MAX) self.featBuf = self.featBuf.slice(-FEAT_MAX);
      self.busy = false;
      if (!running) { self.pump(); return; }
      return classify().then(function (score) {
        updateScore(score);
        self.pump();
      });
    });
  };

  /* ---- UI ------------------------------------------------------------ */

  function updateScore(score) {
    var pct = Math.max(0, Math.min(1, score)) * 100;
    el.bar.style.width = pct + "%";
    el.score.textContent = score.toFixed(3);
    var now = Date.now();
    if (score >= THRESHOLD && now >= cooldownUntil) {
      gateCount++;
      if (gateCount >= GATE_FRAMES) {
        detections++;
        el.count.textContent = detections;
        el.card.classList.add("detected");
        setTimeout(function () { el.card.classList.remove("detected"); }, 600);
        gateCount = 0;
        cooldownUntil = now + COOLDOWN_MS;
      }
    } else if (now >= cooldownUntil) {
      gateCount = 0;
    }
  }

  function toInt16(floatSamples) {
    var out = new Int16Array(floatSamples.length);
    for (var i = 0; i < floatSamples.length; i++)
      out[i] = Math.max(-32768, Math.min(32767, Math.round(floatSamples[i] * 32767)));
    return out;
  }

  function resampleTo16k(floatSamples, fromRate) {
    if (fromRate === 16000) return floatSamples;
    var ratio = fromRate / 16000;
    var out = new Float32Array(Math.floor(floatSamples.length / ratio));
    for (var i = 0; i < out.length; i++) {
      var pos = i * ratio, lo = Math.floor(pos), frac = pos - lo;
      var a = floatSamples[lo] || 0, b = floatSamples[lo + 1] || a;
      out[i] = a + (b - a) * frac;
    }
    return out;
  }

  function downloadModel() {
    el.start.disabled = true;
    if (!ort) {
      if (!window.ort) throw new Error("onnxruntime-web failed to load");
      ort = window.ort;
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.wasmPaths = ORT_BASE;
    }
    var loaded = {}; // key -> ArrayBuffer
    var receivedTotal = 0, totalBytes = 0;
    document.getElementById("progress").style.display = "block";
    var progress = function () {
      var pct = totalBytes ? Math.min(100, Math.round((receivedTotal / totalBytes) * 100)) : 0;
      el.progress.style.width = pct + "%";
      el.start.textContent = "Downloading… " + pct + "%";
      setStatus("Loading model… " + pct + "% (" + Math.round(receivedTotal / 1024) + " of "
        + Math.round(totalBytes / 1024) + " KB)");
    };
    return Promise.all(MODEL_FILES.map(function (file) {
      return fetch(file.url).then(function (response) {
        if (!response.ok) throw new Error("HTTP " + response.status + " for " + file.url);
        totalBytes += Number(response.headers.get("content-length")) || 0;
        progress();
        if (!response.body) return response.arrayBuffer().then(function (b) { loaded[file.key] = b; });
        var reader = response.body.getReader();
        var chunks = [], received = 0;
        function pump() {
          return reader.read().then(function (result) {
            if (result.done) {
              var out = new Uint8Array(received), offset = 0;
              for (var i = 0; i < chunks.length; i++) { out.set(chunks[i], offset); offset += chunks[i].length; }
              loaded[file.key] = out.buffer;
              receivedTotal += received;
              progress();
              return null;
            }
            chunks.push(result.value);
            received += result.value.length;
            receivedTotal += result.value.length;
            progress();
            return pump();
          });
        }
        return pump();
      });
    })).then(function () {
      progress();
      setStatus("Priming…");
      return Promise.all([
        ort.InferenceSession.create(loaded.emb, { executionProviders: ["wasm"] }),
        ort.InferenceSession.create(loaded.cls, { executionProviders: ["wasm"] }),
      ]).then(function (s) {
        sessions = { emb: s[0], cls: s[1] };
        melWeights = new Float32Array(loaded.mel);
        document.getElementById("progress").style.display = "none";
        pipeline = new Pipeline();
        return pipeline.prime();
      });
    }).then(function () {
      el.start.disabled = false;
      el.start.textContent = "Start listening";
      setStatus("Model ready — allow the microphone, then say “yambo”.");
    }).catch(function (error) {
      setStatus("Could not load the model: " + (error.message || error) + ".");
      el.progress.style.width = "0%";
      el.start.disabled = false;
      throw error;
    });
  }

  function startListening() {
    el.start.disabled = true;
    setStatus("Requesting microphone…");
    navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    }).then(function (mic) {
      stream = mic;
      audio = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      var source = audio.createMediaStreamSource(stream);
      processor = audio.createScriptProcessor(2048, 1, 1);
      mute = audio.createGain();
      mute.gain.value = 0;
      processor.onaudioprocess = function (event) {
        if (!running) return;
        var input = event.inputBuffer.getChannelData(0);
        var int16 = toInt16(resampleTo16k(input, audio.sampleRate));
        tail.push.apply(tail, int16);
        while (tail.length >= CHUNK) {
          pending.push(Array.prototype.slice.call(tail, 0, CHUNK));
          tail = tail.slice(CHUNK);
        }
      };
      source.connect(processor);
      processor.connect(mute);
      mute.connect(audio.destination);
      running = true;
      el.start.textContent = "Listening… (click to stop)";
      el.start.disabled = false;
      setStatus("Say “yambo”.");
      timer = setInterval(function () {
        while (pending.length) pipeline.feed(pending.shift());
      }, 80);
    }).catch(function (error) {
      setStatus("Could not access the microphone: " + (error.message || error) + ".");
      el.start.disabled = false;
    });
  }

  function stop() {
    running = false;
    if (timer) { clearInterval(timer); timer = null; }
    pending = [];
    tail = [];
    if (processor) { processor.disconnect(); processor = null; }
    if (mute) { mute.disconnect(); mute = null; }
    if (audio) { audio.close(); audio = null; }
    if (stream) { stream.getTracks().forEach(function (t) { t.stop(); }); stream = null; }
    el.start.textContent = "Start listening";
    el.start.disabled = false;
    el.bar.style.width = "0%";
    setStatus("Model ready.");
  }

  function init() {
    el.card = q("wake-demo");
    el.start = q("wake-demo-start");
    el.status = q("wake-demo-status");
    el.progress = q("wake-demo-progress-bar");
    el.bar = q("wake-demo-bar");
    el.score = q("wake-demo-score");
    el.count = q("wake-demo-count");
    el.start.addEventListener("click", function () {
      if (running) { stop(); return; }
      if (!sessions) {
        el.start.textContent = "Loading…";
        try {
          downloadModel().then(startListening).catch(function () {});
        } catch (error) {
          setStatus("Could not start: " + (error.message || error));
          el.start.textContent = "Download model";
          el.start.disabled = false;
        }
        return;
      }
      startListening();
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
