"""
pos_tagger.py
=============
O(1) unigram POS tagger backed by the Brown Corpus dictionary.

Tagging strategy (in priority order):
  1. Direct lookup in Brown unigram dict (lowercase).
  2. Capitalized token not at sentence start → PROPN (likely proper noun).
  3. Token is all-digits or looks like a number → NUM.
  4. Default fallback → NOUN.

Tags returned use the simplified *Universal POS tagset* subset expected
by chunk_grammar.py:
  NOUN, PROPN, VERB, ADJ, ADV, DET, ADP, NUM, CONJ, PRT, PRON, .
"""

from __future__ import annotations

import re
from ner.data.brown_pos import get_unigram_dict

# Brown corpus Universal tags we keep as-is
_PASS_THROUGH = {"NOUN", "VERB", "ADJ", "ADV", "DET", "ADP", "NUM", "CONJ", "PRT", "PRON", "."}

# Map Penn Treebank tags that might appear in the dict to Universal equivalents
# (Brown corpus already uses Universal when we request tagset='universal', but
#  just in case the cache was built differently we normalise here)
_PTB_TO_UNIV: dict[str, str] = {
    "NN":   "NOUN",  "NNS":  "NOUN",
    "NNP":  "PROPN", "NNPS": "PROPN",
    "VB":   "VERB",  "VBD":  "VERB",  "VBG": "VERB",
    "VBN":  "VERB",  "VBP":  "VERB",  "VBZ": "VERB",
    "JJ":   "ADJ",   "JJR":  "ADJ",   "JJS": "ADJ",
    "RB":   "ADV",   "RBR":  "ADV",   "RBS": "ADV",
    "DT":   "DET",   "PDT":  "DET",
    "IN":   "ADP",   "TO":   "ADP",
    "CD":   "NUM",
    "CC":   "CONJ",
    "RP":   "PRT",   "MD":   "VERB",
    "PRP":  "PRON",  "PRP$": "PRON",  "WP": "PRON", "WP$": "PRON",
    "EX":   "PRON",
    ".":    ".",     ",":    ".",     ":":  ".",
    "WRB":  "ADV",   "WDT":  "DET",
}

_NUMBER_RE = re.compile(r"^\d+(?:[,\.]\d+)*$")


def _normalise_tag(tag: str) -> str:
    """Convert any tag to a Universal-like tag used by the chunker."""
    if tag in _PASS_THROUGH:
        return tag
    return _PTB_TO_UNIV.get(tag, "NOUN")


class UnigramTagger:
    """Stateless O(1)-per-token POS tagger using a unigram dictionary."""

    def __init__(self) -> None:
        self._dict: dict[str, str] = {}

    def load(self) -> "UnigramTagger":
        raw = get_unigram_dict()
        self._dict = {word: _normalise_tag(tag) for word, tag in raw.items()}
        return self

    def tag_token(self, token: str, position: int = 1) -> str:
        """
        Tag a single token.

        Parameters
        ----------
        token:
            Raw token string.
        position:
            Index in the sentence (0-indexed). Position 0 is sentence-start
            and should NOT automatically be tagged PROPN even if capitalised.
        """
        lower = token.lower()

        # Number check
        if _NUMBER_RE.match(token):
            return "NUM"

        # Dict lookup
        tag = self._dict.get(lower)
        if tag:
            return tag

        # Capitalised and not at sentence start → likely proper noun
        if token[0].isupper() and position > 0:
            return "PROPN"

        # Punctuation
        if token in {".", ",", ":", ";", "!", "?", "-", "--"}:
            return "."

        return "NOUN"

    def tag(self, tokens: list[str]) -> list[tuple[str, str]]:
        """Tag a list of tokens, returning (token, tag) pairs."""
        return [(tok, self.tag_token(tok, i)) for i, tok in enumerate(tokens)]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_TAGGER: UnigramTagger | None = None


def get_tagger() -> UnigramTagger:
    global _TAGGER  # noqa: PLW0603
    if _TAGGER is None:
        _TAGGER = UnigramTagger().load()
    return _TAGGER
