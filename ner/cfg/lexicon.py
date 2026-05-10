"""
lexicon.py
==========
Build the terminal-node lexicon for the CFG parser.

The lexicon maps each token string to one or more POS terminal tags.
Priority order:
  1. Special regex terminals (TIME_TERMINAL, MON_TERMINAL) — detected before lookup.
  2. Gazetteer entries  → NNP
  3. Brown Corpus unigram dict
  4. Unknown word fallback (capitalised → NNP, else NN)

Internally, Universal POS tags from Brown are mapped to Penn Treebank–style
tags used in the CNF grammar (e.g. NOUN → NN, PROPN → NNP, DET → DT …).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ner.data.brown_pos import get_unigram_dict
from ner.data.gazetteers import get_gazetteers
from ner.patterns.regex_patterns import TIME_PATTERN, MONETARY_PATTERN

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Special terminal symbols injected BEFORE normal lexical lookup
# ---------------------------------------------------------------------------
TIME_TERMINAL = "TIME_T"
MON_TERMINAL  = "MON_T"

# ---------------------------------------------------------------------------
# Universal → Penn Treebank terminal tag mapping
# ---------------------------------------------------------------------------
_UNIV_TO_PTB: dict[str, str] = {
    "NOUN":  "NN",
    "VERB":  "VB",
    "ADJ":   "JJ",
    "ADV":   "RB",
    "PRON":  "PRP",
    "DET":   "DT",
    "ADP":   "IN",
    "NUM":   "CD",
    "CONJ":  "CC",
    "PRT":   "RP",
    ".":     ".",
    "X":     "NN",   # treat unknowns as common noun
}

_DATA_DIR = Path(__file__).parent.parent / "data"
_CACHE_PATH = _DATA_DIR / "cfg_lexicon.json"


class Lexicon:
    """Maps token → list[terminal_tag] for CFG parsing."""

    def __init__(self) -> None:
        self._word_to_tag: dict[str, str] = {}
        self._location_set: frozenset[str] = frozenset()
        self._person_set: frozenset[str] = frozenset()

    # ------------------------------------------------------------------
    def build(self, force: bool = False) -> "Lexicon":
        """Populate the lexicon from Brown + Gazetteers."""
        _DATA_DIR.mkdir(parents=True, exist_ok=True)

        if _CACHE_PATH.exists() and not force:
            logger.info("Loading cached CFG lexicon from %s", _CACHE_PATH)
            with open(_CACHE_PATH, encoding="utf-8") as fh:
                data = json.load(fh)
            self._word_to_tag = data["word_to_tag"]
            self._location_set = frozenset(data["locations"])
            self._person_set = frozenset(data["persons"])
            return self

        logger.info("Building CFG lexicon…")

        # Layer 1 – Brown unigram dict
        unigram = get_unigram_dict()
        for word, univ_tag in unigram.items():
            ptb_tag = _UNIV_TO_PTB.get(univ_tag, "NN")
            self._word_to_tag[word.lower()] = ptb_tag

        # Layer 2 – Gazetteer phrase sets (used only in disambiguate(), NOT for
        # sub-token NNP expansion, to avoid common English words like 'is',
        # 'new', 'city' being wrongly tagged NNP because they appear in city names)
        gaz = get_gazetteers()
        locations = frozenset(gaz["location"])
        persons   = frozenset(gaz["person"])
        # We intentionally do NOT expand sub-tokens here.

        self._location_set = locations
        self._person_set   = persons

        with open(_CACHE_PATH, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "word_to_tag": self._word_to_tag,
                    "locations":   sorted(locations),
                    "persons":     sorted(persons),
                },
                fh, ensure_ascii=False,
            )
        logger.info("CFG lexicon built: %d entries", len(self._word_to_tag))
        return self

    # ------------------------------------------------------------------
    def tag_token(self, token: str) -> str:
        """Return the best terminal tag for a single token (no regex check).

        Capitalisation heuristic (applied BEFORE Brown lookup):
          If a token starts with an uppercase letter, it is very likely a
          proper noun. Brown corpus frequently tags proper names as NN since
          it operates on lowercase. We override NN→NNP for such tokens.
        """
        if not token:
            return "NN"
        # Capitalisation override: uppercase-starting tokens treated as NNP
        # UNLESS the token is all-uppercase short word (acronym check later)
        # or it's a known function word in Brown (DT, VB, IN, etc.)
        tag = self._word_to_tag.get(token.lower())
        if token[0].isupper():
            # Keep non-noun tags (VB, DT, IN, CC, RB, PRP, …) from Brown
            # but override NN → NNP for proper-looking tokens
            if tag is None or tag in ("NN", "NNS"):
                return "NNP"
            return tag  # return Brown tag if it's something useful (VB, DT…)
        # Lowercase token
        if tag:
            return tag
        return "NN"

    def tag_sentence(self, sentence: str) -> list[tuple[str, str]]:
        """
        Tag every token in *sentence*, substituting regex-matched Time/Monetary
        spans with their special terminal symbols first.

        Tokenisation splits on whitespace AND detaches leading/trailing
        punctuation (e.g. 'London.' → 'London' '.')

        Returns
        -------
        list of (token_text, terminal_tag)
        """
        import re
        tagged: list[tuple[str, str]] = []
        pos = 0
        # Merge and sort all special spans
        spans: list[tuple[int, int, str]] = []
        for m in TIME_PATTERN.finditer(sentence):
            spans.append((m.start(), m.end(), TIME_TERMINAL))
        for m in MONETARY_PATTERN.finditer(sentence):
            spans.append((m.start(), m.end(), MON_TERMINAL))
        spans.sort(key=lambda x: x[0])

        # Remove overlapping spans (keep first)
        clean_spans: list[tuple[int, int, str]] = []
        last_end = -1
        for s, e, tag in spans:
            if s >= last_end:
                clean_spans.append((s, e, tag))
                last_end = e

        def _tokenise_gap(text: str) -> list[str]:
            """Whitespace-split and detach leading/trailing punctuation."""
            tokens: list[str] = []
            for raw in re.split(r"\s+", text.strip()):
                if not raw:
                    continue
                # Strip leading punctuation
                while raw and raw[0] in '.,;:!?"\'-()[]{}—–':
                    tokens.append(raw[0])
                    raw = raw[1:]
                if not raw:
                    continue
                # Strip trailing punctuation
                trail: list[str] = []
                while raw and raw[-1] in '.,;:!?"\'-()[]{}—–':
                    trail.append(raw[-1])
                    raw = raw[:-1]
                if raw:
                    tokens.append(raw)
                tokens.extend(reversed(trail))
            return tokens

        for start, end, special_tag in clean_spans:
            # Tag the gap before this span
            for tok in _tokenise_gap(sentence[pos:start]):
                tagged.append((tok, self.tag_token(tok)))
            # Inject special terminal
            tagged.append((sentence[start:end], special_tag))
            pos = end

        # Tail of sentence
        for tok in _tokenise_gap(sentence[pos:]):
            tagged.append((tok, self.tag_token(tok)))

        return tagged


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_LEXICON: Lexicon | None = None


def get_lexicon() -> Lexicon:
    global _LEXICON  # noqa: PLW0603
    if _LEXICON is None:
        _LEXICON = Lexicon().build()
    return _LEXICON
