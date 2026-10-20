from wakesmith.config import Args
from wakesmith.evaluation import model_card, report


def test_model_card_and_report(tmp_path):
    evaluation = {
        "thresholds": ["0.5", "0.7"],
        "recall": {"kokoro_unseen_voices/clean": {"0.5": 0.9, "0.7": 0.8}},
        "negative_streams": {
            "english": {"hours": 5.3, "false_activations_per_hour": {"0.5": 0.0, "0.7": 0.0}}
        },
    }
    card = model_card(Args.resolve("tomtom"), "abc", 0.5, evaluation, {"0.5": {"recall": 0.9}})
    assert card["recommended"]["threshold"] == 0.5
    assert card["held_out"]["recall"]["kokoro_unseen_voices/clean"] == 0.9
    report(
        card,
        evaluation,
        {"kokoro_positive_train": {"generated": 10, "accepted": 8}},
        tmp_path / "r.md",
    )
    text = (tmp_path / "r.md").read_text()
    assert "90.0%" in text and "| kokoro_positive_train | 10 | 8 |" in text
