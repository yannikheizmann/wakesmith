"""Decide whether ASR transcripts were heard as the wake word."""

from __future__ import annotations

import re

from rapidfuzz.distance import Levenshtein

from ._ipa import fold, phonemize

VOWELS = frozenset("aeiouyæɑɒɔəɚɛɜɪʊʌøœɐɤɯ")
FILLERS = frozenset({"a", "ah", "ey", "hey", "hi", "hallo", "oh", "ok", "okay", "o.k.", "ja"})


def split(folded: str) -> tuple[str, str]:
    """Split folded IPA into its consonant skeleton and its vowel sequence."""
    return (
        "".join(c for c in folded if c not in VOWELS),
        "".join(c for c in folded if c in VOWELS),
    )


def normalise(text: str) -> str:
    """Lowercase, drop punctuation and remove leading filler words."""
    words = re.sub(r"[^\w' ]+", " ", text.lower()).split()
    while len(words) > 1 and words[0] in FILLERS:
        words = words[1:]
    return " ".join(words)


class Matcher:
    """Phonetic acceptance test for transcripts of synthetic clips.

    A transcript matches when, after leading fillers are removed, its consonant skeleton
    equals the wake word's and its vowels differ by at most ``vowel_tolerance`` (normalised
    edit distance). Vowel quality varies across voices and accents; a changed, missing or
    extra consonant (``tomcom``, ``omtom`` or ``tom tom tom`` for ``tomtom``) makes another word.
    A trailing ``s``/``z`` is tolerated because ASR often hears a release burst as a plural.
    """

    def __init__(self, wake_word: str, language: str = "en-us", vowel_tolerance: float = 0.5):
        self.language = language
        self.consonants, self.vowels = split(fold(phonemize([wake_word], language)[0]))
        self.vowel_tolerance = vowel_tolerance

    def matches(self, folded: str) -> bool:
        consonants, vowels = split(folded)
        plural = consonants[:-1] == self.consonants and consonants[-1:] in ("s", "z")
        if consonants != self.consonants and not plural:
            return False
        longest = max(len(vowels), len(self.vowels), 1)
        return Levenshtein.distance(vowels, self.vowels) / longest <= self.vowel_tolerance

    def __call__(self, transcripts: list[str]) -> list[bool]:
        texts = [normalise(t) for t in transcripts]
        ipa = phonemize([t or "_" for t in texts], self.language)
        return [bool(t) and self.matches(fold(p)) for t, p in zip(texts, ipa, strict=True)]
