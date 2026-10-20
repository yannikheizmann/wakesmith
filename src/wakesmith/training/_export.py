"""Export the classifier to openWakeWord-compatible ONNX."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from wakesmith.config import CONTEXT_FRAMES, EMBEDDING_DIM


def export_onnx(model: torch.nn.Module, path: Path) -> None:
    """Write ``(batch, 16, 96) -> (batch, 1)`` ONNX and verify it against PyTorch."""
    import onnxruntime as ort

    model = model.cpu().eval()
    partial = path.with_name(path.stem + ".partial.onnx")
    torch.onnx.export(
        model,
        torch.randn(1, CONTEXT_FRAMES, EMBEDDING_DIM),
        str(partial),
        opset_version=17,
        dynamo=False,
        input_names=["embeddings"],
        output_names=["score"],
        dynamic_axes={"embeddings": {0: "batch"}, "score": {0: "batch"}},
    )
    session = ort.InferenceSession(str(partial), providers=["CPUExecutionProvider"])
    x = (
        np.random.default_rng(0)
        .standard_normal((3, CONTEXT_FRAMES, EMBEDDING_DIM))
        .astype(np.float32)
    )
    np.testing.assert_allclose(
        session.run(None, {"embeddings": x})[0],
        model(torch.from_numpy(x)).detach().numpy(),
        rtol=1e-3,
        atol=1e-5,
    )
    partial.replace(path)
