import numpy as np
import pytest
import torch

from wakesmith.config import Args
from wakesmith.training import TrainingData, export_onnx, mine_hard_negatives, train, validate


@pytest.fixture
def args() -> Args:
    return Args.resolve(
        "tomtom",
        training={
            "steps": 40,
            "hard_negatives": 64,
            "model_size": "small",
            "batch": {
                "kokoro": 32,
                "voxcpm": 0,
                "adversarial": 32,
                "acav": 64,
                "speech": 16,
                "background": 16,
                "hard": 16,
            },
        },
    )


def test_batches_mix_all_pools(feature_dir, args):
    data = TrainingData(*feature_dir, args, torch.device("cpu"))
    x, y = data.next()
    assert x.shape == (32 + 32 + 16 + 64 + 16, 16, 96)
    assert y.sum() == 32
    assert data.acav_validation.shape[0] == 4096 - int(4096 * 0.95)


def test_training_learns_separable_data_and_exports(feature_dir, args, tmp_path):
    data = TrainingData(*feature_dir, args, torch.device("cpu"))
    result = train(args, data)
    assert set(result.metrics) >= {"0.5", "0.9"}
    assert result.metrics["0.5"]["recall"] > 0.9
    assert result.metrics["0.5"]["adversarial_accept"] < 0.1
    assert any("hard_negative_mining" in entry for entry in result.log)
    path = tmp_path / "tomtom.onnx"
    export_onnx(result.model, path)
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape[1:] == [16, 96]
    score = session.run(None, {"embeddings": np.zeros((2, 16, 96), np.float32)})[0]
    assert score.shape == (2, 1)


def test_mining_keeps_the_highest_scores(feature_dir, args):
    from wakesmith.training import build_model

    data = TrainingData(*feature_dir, args, torch.device("cpu"))
    summary = mine_hard_negatives(build_model(args), data, n=50)
    assert summary["count"] == 50 and data.hard.shape == (50, 16, 96)
    assert summary["max_score"] >= summary["min_score"]
    assert set(validate(build_model(args), data)["0.5"]["false_activations_per_hour"]) == {
        "openwakeword_validation",
        "speech_validation",
        "acav_heldout_windows",
    }
