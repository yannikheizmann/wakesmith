import onnxruntime
import gradio as gr
import numpy as np

"""Live wake-word demo for the "yambo" model trained with wakesmith.

The streaming pipeline mirrors openWakeWord's AudioFeatures/Model exactly and was
verified numerically against openwakeword 0.6 (max score difference 9e-8).
"""

THRESHOLD = 0.5
GATE_FRAMES = 2
COOLDOWN_FRAMES = 25  # 2 s at 80 ms frames
RAW_MAX = 160_000
MEL_MAX = 970
FEAT_MAX = 120
CONTEXT_FRAMES = 480
CHUNK = 1280

mel_session = onnxruntime.InferenceSession("models/melspectrogram.onnx", providers=["CPUExecutionProvider"])
embedding_session = onnxruntime.InferenceSession("models/embedding_model.onnx", providers=["CPUExecutionProvider"])
classifier_session = onnxruntime.InferenceSession("models/yambo.onnx", providers=["CPUExecutionProvider"])


def _mel(x):
    out = mel_session.run(None, {"input": np.asarray(x, np.float32)[None, :]})[0]
    return (np.squeeze(out) / 10 + 2).astype(np.float32)


def _embedding(mel_buffer):
    window = mel_buffer[-76:].astype(np.float32)[None, :, :, None]
    return embedding_session.run(None, {"input_1": window})[0].reshape(-1)


def prime_buffer(noise):
    """openWakeWord's reset(): embeddings over 4 s of random noise."""
    spec = _mel(noise)
    windows = [spec[i:i + 76] for i in range(0, spec.shape[0], 8) if spec[i:i + 76].shape[0] == 76]
    batch = np.expand_dims(np.array(windows), axis=-1).astype(np.float32)
    return embedding_session.run(None, {"input_1": batch})[0].reshape(len(windows), -1)


class Pipeline:
    """The exact streaming state of the deployed runtime."""

    def __init__(self, prime_noise):
        self.raw = []
        self.mel_buf = np.ones((76, 32), np.float32)
        self.accumulated = 0
        self.remainder = np.empty(0, np.int16)
        self.feat_buf = prime_buffer(prime_noise).astype(np.float32)

    def feed(self, chunk):
        x = np.concatenate((self.remainder, chunk)) if self.remainder.size else chunk
        self.remainder = np.empty(0, np.int16)
        if self.accumulated + x.shape[0] >= CHUNK:
            rem = (self.accumulated + x.shape[0]) % CHUNK
            if rem != 0:
                even, self.remainder = x[:-rem], x[-rem:]
                self.raw.extend(even.tolist())
                self.accumulated += len(even)
            else:
                self.raw.extend(x.tolist())
                self.accumulated += x.shape[0]
        else:
            self.accumulated += x.shape[0]
            self.raw.extend(x.tolist())

        if self.accumulated >= CHUNK and self.accumulated % CHUNK == 0:
            take = min(len(self.raw), self.accumulated + CONTEXT_FRAMES)
            self.mel_buf = np.vstack((self.mel_buf, _mel(np.array(self.raw[-take:], np.int16))))
            if self.mel_buf.shape[0] > MEL_MAX:
                self.mel_buf = self.mel_buf[-MEL_MAX:, :]
            for i in range(self.accumulated // CHUNK - 1, -1, -1):
                ndx = -8 * i if i != 0 else self.mel_buf.shape[0]
                window = self.mel_buf[-76 + ndx:ndx]
                if window.shape[0] == 76:
                    self.feat_buf = np.vstack((self.feat_buf, _embedding(window)))
            self.accumulated = 0
        if self.feat_buf.shape[0] > FEAT_MAX:
            self.feat_buf = self.feat_buf[-FEAT_MAX:, :]

    def score(self):
        return float(classifier_session.run(
            None, {"embeddings": self.feat_buf[-16:, :][None, :]})[0].reshape(-1)[-1])


def to_int16(samples):
    return (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)


def resample_to_16k(samples, from_rate):
    if from_rate == 16000:
        return samples
    ratio = from_rate / 16000
    return np.interp(np.arange(0, len(samples), ratio),
                     np.arange(len(samples)), samples).astype(np.float32)


def wave(chunk, state):
    """One streaming microphone callback: feed the chunk, update the meter."""
    sr, data = chunk if isinstance(chunk, tuple) else (16000, chunk)
    if state is None or "pipeline" not in state:
        noise = np.random.RandomState(0).randint(-1000, 1000, 16000 * 4).astype(np.int16)
        state = {"pipeline": Pipeline(noise), "count": 0, "gate": 0, "cooldown": 0}
    pipeline = state["pipeline"]

    chunk16 = to_int16(resample_to_16k(np.asarray(data, np.float32).reshape(-1), sr))
    if chunk16.size:
        pipeline.feed(chunk16)
    score = pipeline.score()

    frame_gate = score >= THRESHOLD
    state["gate"] = state["gate"] + 1 if frame_gate else 0
    if state["gate"] >= GATE_FRAMES and state["cooldown"] <= 0:
        state["count"] += 1
        state["cooldown"] = COOLDOWN_FRAMES
    if state["cooldown"] > 0:
        state["cooldown"] -= 1
    indicator = "🔔 detected!" if state["cooldown"] > 0 and state["gate"] == 0 else (
        "listening… say “yambo”" if not frame_gate else "…")

    return {"yambo": round(score, 3), "other": round(1 - score, 3)}, state["count"], indicator, state


with gr.Blocks(title='Say "yambo"') as demo:
    gr.Markdown(
        "## Say **“yambo”**\n"
        "A wake-word model trained with "
        "[wakesmith](https://github.com/yannikheizmann/wakesmith) — the score comes "
        "from the real openWakeWord streaming pipeline. Everything runs on this "
        "Space's CPU; nothing is stored."
    )
    mic = gr.Audio(sources=["microphone"], streaming=True, label="microphone", type="numpy")
    meter = gr.Label(num_top_classes=2, label="model score — threshold 0.5")
    with gr.Row():
        count = gr.Number(label="detections", precision=0)
        status = gr.Textbox(label="detector", value="listening… say “yambo”")
    state = gr.State(value=None)
    mic.stream(wave, inputs=[mic, state], outputs=[meter, count, status, state])

if __name__ == "__main__":
    demo.launch()
