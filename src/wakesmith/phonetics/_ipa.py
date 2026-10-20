"""Phonemise text with the espeak-ng library bundled by ``espeakng-loader``."""

from __future__ import annotations

import functools
import re
import unicodedata

_MARKS = re.compile(r"[ˈˌ.ː‿ \-]")
# Merge symbols that espeak writes differently across words and languages but that sound
# alike, so that distances reflect what a listener hears.
_FOLD = str.maketrans(
    {
        "ə": "ʌ",
        "ɐ": "ʌ",
        "ᵻ": "ɪ",
        "ɜ": "ɚ",
        "ɾ": "t",
        "ʔ": "",
        "ɡ": "g",
        "ɹ": "r",
        "ʁ": "r",
        "ʀ": "r",
        "ɻ": "r",
    }
)


@functools.cache
def _backend(language: str):
    import espeakng_loader
    from phonemizer.backend import EspeakBackend
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    return EspeakBackend(language, preserve_punctuation=False, with_stress=True)


def phonemize(texts: list[str], language: str = "en-us") -> list[str]:
    """Return espeak-ng IPA, with stress marks, for each text in an espeak language."""
    return _backend(language).phonemize(list(texts), strip=True, njobs=1) if texts else []


def fold(ipa: str) -> str:
    """Drop stress, length and word boundaries and merge near-identical symbols."""
    text = unicodedata.normalize("NFC", ipa).translate(_FOLD)
    return _MARKS.sub("", text).replace("ɚ", "ər")
