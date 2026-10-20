import pytest

from wakesmith.phonetics import Matcher, confusables, fold, fragments, phonemize


def test_fold_removes_stress_and_merges_symbols():
    assert fold("tˈɑːmtəm") == fold("tɑmtˌʌm")


@pytest.mark.parametrize(
    ("transcript", "expected"),
    [
        ("Tomtom.", True),
        ("Tom Tom!", True),
        ("Hey, Tomtom.", True),
        ("Tumtum", True),
        ("Tomtoms", True),
        ("Tomcom", False),
        ("Bomtom", False),
        ("Tom", False),
        ("Tom Tom Tom", False),
        ("", False),
    ],
)
def test_matcher_requires_the_consonant_skeleton(transcript, expected):
    assert Matcher("tomtom")([transcript]) == [expected]


def test_matcher_in_german():
    assert Matcher("Computer", "de")(["Computer!", "Kompost"]) == [True, False]


def test_fragments_cover_partial_utterances():
    assert "jarvis" in fragments("hey jarvis")
    assert fragments("computer") == ["compu", "uter"]


def test_confusables_never_include_words_heard_as_the_wake_word():
    phrases = confusables("tomtom", extra=("tom thumb",))
    assert "tom thumb" in phrases and "tom" in phrases
    assert not any(Matcher("tomtom")(sorted(set(phrases))))
    assert not any(" um" in p or p.startswith("um ") for p in phrases)


def test_phonemize_returns_one_result_per_text():
    assert len(phonemize(["one", "two"])) == 2
