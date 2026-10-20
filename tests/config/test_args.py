import pytest
from pydantic import ValidationError

from wakesmith.config import Args


def test_defaults_reproduce_the_standard_recipe():
    args = Args.resolve("tomtom")
    assert args.kokoro.n_train == 24000 and args.voxcpm.enabled and args.training.steps == 60000


def test_yaml_then_overrides_take_precedence(tmp_path):
    config = tmp_path / "args.yaml"
    config.write_text("training:\n  steps: 123\nkokoro:\n  n_train: 7\n")
    args = Args.resolve("tomtom", config, training={"steps": 456})
    assert args.kokoro.n_train == 7
    assert args.training.steps == 456
    assert args.training.learning_rate == Args.resolve("tomtom").training.learning_rate


def test_any_change_produces_a_new_digest():
    assert Args.resolve("tomtom").digest != Args.resolve("tomtom", training={"steps": 1}).digest
    assert Args.resolve("tomtom").digest == Args.resolve(" tomtom ").digest


def test_non_english_words_add_native_negative_speech():
    assert Args.resolve("Computer", language="de").negative_speech.language == "de"
    assert Args.resolve("tomtom").negative_speech.language is None
    explicit = Args.resolve("tomtom", negative_speech={"language": "de"})
    assert explicit.negative_speech.language == "de"


@pytest.mark.parametrize("word", ["", "x", "1234", "a" * 60])
def test_rejects_unusable_wake_words(word):
    with pytest.raises(ValidationError):
        Args.resolve(word)


def test_rejects_unknown_fields_and_languages():
    with pytest.raises(ValidationError):
        Args.resolve("tomtom", language="xx")
    with pytest.raises(ValidationError):
        Args.resolve("tomtom", training={"unknown": 1})
    with pytest.raises(ValidationError):
        Args.resolve("tomtom", negative_speech={"language": "en"})


def test_slug_and_phrases():
    args = Args.resolve("Hey Jarvis")
    assert args.slug == "hey_jarvis"
    assert "Hey, Hey Jarvis." not in args.positive_phrases()
    assert "Hey Jarvis!" in args.positive_phrases()
