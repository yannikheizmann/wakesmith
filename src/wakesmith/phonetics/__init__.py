"""Phonetic matching and confusable-phrase search in any espeak-ng language."""

from ._ipa import fold, phonemize
from ._matcher import Matcher
from ._neighbours import confusables, fragments, neighbours

__all__ = ["Matcher", "confusables", "fold", "fragments", "neighbours", "phonemize"]
