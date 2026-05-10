"""
chunk_grammar.py
================
POS-tag regex rules for the Shallow Chunker.

The grammar is expressed as a string of rules in NLTK RegexpParser format
(even though we also ship a hand-coded FSM equivalent for zero-dependency use).

Tag inventory (Universal POS tagset):
  NOUN   – common noun
  PROPN  – proper noun (we map NLTK's NNP/NNPS → PROPN)
  ADJ    – adjective
  DET    – determiner
  ADP    – adposition (preposition)
  NUM    – numeral
  VERB   – verb
  ADV    – adverb
  CONJ   – conjunction
  PRON   – pronoun
  PRT    – particle
  .      – punctuation

Rules (in priority order – first match wins):
  1. Proper-noun run     e.g. "New York", "Edward Jones", "University of Technology"
  2. NP with prep bridge e.g. "University of Technology", "Bank of America"
  3. Simple NP           e.g. "the big company", "a person"
  4. Numeric NP          e.g. "three people", "2 companies"
  5. Mixed proper+common e.g. "President Obama"
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# NLTK-style grammar string (used by RegexpParser when NLTK is available)
# ---------------------------------------------------------------------------
CHUNK_GRAMMAR_STR: str = r"""
NP: {<PROPN>+<ADP><DET>?<PROPN|NOUN>+}
NP: {<PROPN>+}
NP: {<DET>?<ADJ>*<PROPN>+<NOUN>*}
NP: {<DET>?<ADJ>*<NOUN>+<ADP><DET>?<NOUN>+}
NP: {<DET>?<ADJ>*<NOUN>+}
NP: {<NUM><NOUN>+}
"""

# ---------------------------------------------------------------------------
# Tag sequences for the hand-coded FSM chunker
# (each rule is a list of tag sets; a tag set = any of those tags matches)
# ---------------------------------------------------------------------------
# A "tag set" is a frozenset of acceptable tags at that position.
# None in a position means "zero or more of the previous set" (Kleene+).
# We keep it simple: each rule is a sequence of (tag_options, repetition)
# pairs where repetition is "1" (exactly one) or "+" (one or more) or "?" (optional).

# This is consumed by chunker.py's hand-coded FSM.
FSM_RULES: list[dict] = [
    # Rule 1: PROPN+ ADP (DET)? (PROPN|NOUN)+   → "University of Technology"
    {
        "name": "PROPN_PREP_PROPN",
        "sequence": [
            {"tags": {"PROPN"}, "repeat": "+"},
            {"tags": {"ADP"},   "repeat": "1"},
            {"tags": {"DET"},   "repeat": "?"},
            {"tags": {"PROPN", "NOUN"}, "repeat": "+"},
        ],
    },
    # Rule 2: PROPN+   → "New York", "Edward"
    {
        "name": "PROPN_RUN",
        "sequence": [
            {"tags": {"PROPN"}, "repeat": "+"},
        ],
    },
    # Rule 3: DET? ADJ* PROPN+ NOUN*   → "the great President Obama"
    {
        "name": "DET_ADJ_PROPN_NOUN",
        "sequence": [
            {"tags": {"DET"},  "repeat": "?"},
            {"tags": {"ADJ"},  "repeat": "*"},
            {"tags": {"PROPN"}, "repeat": "+"},
            {"tags": {"NOUN"}, "repeat": "*"},
        ],
    },
    # Rule 4: DET? ADJ* NOUN+ ADP DET? NOUN+   → "bank of the river"
    {
        "name": "NOUN_PREP_NOUN",
        "sequence": [
            {"tags": {"DET"},  "repeat": "?"},
            {"tags": {"ADJ"},  "repeat": "*"},
            {"tags": {"NOUN"}, "repeat": "+"},
            {"tags": {"ADP"},  "repeat": "1"},
            {"tags": {"DET"},  "repeat": "?"},
            {"tags": {"NOUN"}, "repeat": "+"},
        ],
    },
    # Rule 5: DET? ADJ* NOUN+   → "the big company"
    {
        "name": "SIMPLE_NP",
        "sequence": [
            {"tags": {"DET"},  "repeat": "?"},
            {"tags": {"ADJ"},  "repeat": "*"},
            {"tags": {"NOUN"}, "repeat": "+"},
        ],
    },
    # Rule 6: NUM NOUN+   → "three people"
    {
        "name": "NUM_NOUN",
        "sequence": [
            {"tags": {"NUM"},  "repeat": "1"},
            {"tags": {"NOUN"}, "repeat": "+"},
        ],
    },
]
