# Using the model

A wakesmith model is a standard openWakeWord classifier: it reads the last 16 embedding
frames (1.28 s) and returns a score between 0 and 1.

```python
import numpy as np
from openwakeword.model import Model

model = Model(wakeword_models=["tomtom.onnx"], inference_framework="onnx")
for _ in range(40):  # Prime the buffers with silence, as during training.
    model.predict(np.zeros(1280, dtype=np.int16))


def on_audio(frame: np.ndarray) -> float:
    """Score one 80 ms frame of 16 kHz int16 audio."""
    return model.predict(frame)["tomtom"]
```

## Activation gate

Activate when the score reaches the recommended threshold (in `<word>.json`) on two
consecutive frames, then ignore activations for two seconds. This is the gate the
evaluation uses; a single-frame gate activates more often than the report states.

```python
class Gate:
    def __init__(self, threshold: float, consecutive: int = 2, cooldown: float = 2.0):
        self.threshold, self.consecutive, self.cooldown = threshold, consecutive, cooldown
        self.hits, self.last = 0, float("-inf")

    def update(self, score: float, now: float) -> bool:
        self.hits = self.hits + 1 if score >= self.threshold else 0
        if self.hits >= self.consecutive and now - self.last >= self.cooldown:
            self.last, self.hits = now, 0
            return True
        return False
```

## Integrations

Any software that loads custom openWakeWord ONNX models can use the file directly, including
Home Assistant's openWakeWord add-on and Wyoming satellites. Copy `<word>.onnx` to the
integration's custom model directory and set its threshold to the recommended value.
