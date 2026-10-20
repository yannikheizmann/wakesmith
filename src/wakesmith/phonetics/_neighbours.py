"""Find words and phrases that sound close to, but not like, the wake word."""

from __future__ import annotations

import functools

from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from ._ipa import fold, phonemize

LEXICON_SIZE = 60000


@functools.cache
def lexicon(language: str, espeak: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The most frequent words of a language and their folded pronunciations."""
    from wordfreq import top_n_list

    words = tuple(
        w
        for w in top_n_list(language, LEXICON_SIZE, wordlist="large")
        if w.isalpha() and len(w) > 1
    )
    return words, tuple(fold(p) for p in phonemize(list(words), espeak))


def neighbours(
    wake_word: str,
    language: str = "en",
    espeak: str = "en-us",
    limit: int = 400,
    max_distance: float = 0.5,
) -> list[str]:
    """Rank confusable single words and two-word phrases by phonetic distance.

    Two-word candidates approximate the wake word's sound with the concatenation of two
    lexicon words (``tom`` + ``tom``). Homophones (distance 0) are excluded, and no more than
    three phrases share a first word so that one prefix cannot crowd out the rest.
    """
    target = fold(phonemize([wake_word], espeak)[0])
    names, sounds = lexicon(language, espeak)
    scorer = Levenshtein.normalized_distance

    def nearest(sound: str, cutoff: float, n: int):
        return process.extract(sound, sounds, scorer=scorer, limit=n, score_cutoff=cutoff)

    singles = sorted(
        (score, names[i]) for _, score, i in nearest(target, max_distance, limit) if score > 0
    )
    middle = len(target) // 2
    heads, tails = nearest(target[: middle + 1], 0.4, 20), nearest(target[middle:], 0.4, 20)
    pairs, per_head = [], {}
    for _, _, i in heads:
        for _, _, j in tails:
            if min(len(names[i]), len(names[j])) < 3:  # Skip fillers such as "um".
                continue
            score = scorer(sounds[i] + sounds[j], target)
            if 0 < score <= max_distance:
                pairs.append((score, names[i], f"{names[i]} {names[j]}"))
    kept = []
    for score, head, phrase in sorted(pairs):
        if per_head.get(head, 0) < 3:
            per_head[head] = per_head.get(head, 0) + 1
            kept.append((score, phrase))
    n_single = min(len(singles), int(limit * 0.6))
    ranked = sorted(singles[:n_single] + kept[: limit - n_single])
    seen, phrases = {wake_word.lower()}, []
    for _, phrase in ranked:
        if phrase not in seen:
            seen.add(phrase)
            phrases.append(phrase)
    return phrases


def fragments(wake_word: str) -> list[str]:
    """Partial utterances: dropped words and the leading or trailing half of a long word."""
    words = wake_word.lower().split()
    parts = set(words) if len(words) > 1 else set()
    parts.update(" ".join(words[:i] + words[i + 1 :]) for i in range(len(words)) if len(words) > 1)
    if len(words[-1]) >= 5:
        word = words[-1]
        parts.update({word[: len(word) // 2 + 1], word[len(word) // 2 :]})
    parts.discard(wake_word.lower())
    return sorted(parts)


def confusables(
    wake_word: str,
    extra: tuple[str, ...] = (),
    language: str = "en",
    espeak: str = "en-us",
    limit: int = 400,
) -> list[str]:
    """Adversarial phrase list: neighbours plus fragments and user phrases weighted up.

    Fragments and explicit phrases are the closest confusions, so they are repeated to make
    up roughly a third of all adversarial clips.
    """
    from ._matcher import Matcher

    close = [*fragments(wake_word), *extra]
    found = [p for p in neighbours(wake_word, language, espeak, limit) if p not in close]
    # Inflections such as "computers" would pass the positive screen; never train them as both.
    heard = Matcher(wake_word, espeak)(found)
    found = [p for p, h in zip(found, heard, strict=True) if not h]
    return found + close * max(1, round(len(found) / 2 / max(1, len(close))))
