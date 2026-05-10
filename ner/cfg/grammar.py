"""
grammar.py
==========
Minimal Context-Free Grammar in Chomsky Normal Form (CNF) for NER.

The grammar focuses exclusively on structures relevant to extracting
Named Entities (NP, PP, TIME_PHRASE, MONETARY_PHRASE) without attempting
to cover all of English syntax (which would require ~72K Penn Treebank rules).

Terminal symbols
----------------
  NN, NNP, NNPS, NNS  – noun tags
  VB, VBZ, VBD, VBG, VBN, VBP – verb tags
  JJ, JJR, JJS – adjective tags
  DT            – determiner
  IN            – preposition/conjunction
  CC            – coordinating conjunction
  CD            – cardinal number
  PRP, PRP$     – pronoun
  RP, RB        – particle, adverb
  .  ,  :       – punctuation
  TIME_T        – injected by lexicon for time spans
  MON_T         – injected by lexicon for monetary spans

Non-terminal symbols
--------------------
  S              – sentence (root)
  NP             – noun phrase
  VP             – verb phrase
  PP             – prepositional phrase
  ADJP           – adjective phrase
  ADVP           – adverb phrase
  TIME_PHRASE    – time entity phrase
  MON_PHRASE     – monetary entity phrase

CNF format
----------
Each production is a (lhs, rhs) tuple where rhs is either:
  - (terminal_tag,)          → unary rule mapped to a terminal
  - (non_terminal, non_terminal) → binary rule
"""

from __future__ import annotations

from typing import NamedTuple


class Production(NamedTuple):
    lhs: str        # left-hand side non-terminal
    rhs: tuple      # (terminal_tag,) OR (nt1, nt2)


# ---------------------------------------------------------------------------
# Terminal productions  (lhs → terminal_tag)
# ---------------------------------------------------------------------------
TERMINAL_PRODUCTIONS: list[Production] = [
    # Nouns
    Production("NN",   ("NN",)),
    Production("NN",   ("NNS",)),
    Production("NNP",  ("NNP",)),
    Production("NNP",  ("NNPS",)),

    # Special entity terminals
    Production("TIME_PHRASE", ("TIME_T",)),
    Production("MON_PHRASE",  ("MON_T",)),

    # Verbs
    Production("VB",   ("VB",)),
    Production("VB",   ("VBZ",)),
    Production("VB",   ("VBD",)),
    Production("VB",   ("VBG",)),
    Production("VB",   ("VBN",)),
    Production("VB",   ("VBP",)),

    # Adjectives
    Production("JJ",   ("JJ",)),
    Production("JJ",   ("JJR",)),
    Production("JJ",   ("JJS",)),

    # Determiners, prepositions, numbers, pronouns, particles
    Production("DT",   ("DT",)),
    Production("IN",   ("IN",)),
    Production("CD",   ("CD",)),
    Production("PRP",  ("PRP",)),
    Production("PRP",  ("PRP$",)),
    Production("RB",   ("RB",)),
    Production("CC",   ("CC",)),
    Production("RP",   ("RP",)),
    Production("PUNC", (".",)),
    Production("PUNC", (",",)),
    Production("PUNC", (":",)),
]

# ---------------------------------------------------------------------------
# Binary productions  (lhs → nt1 nt2)
# ---------------------------------------------------------------------------
BINARY_PRODUCTIONS: list[Production] = [

    # --- Noun Phrase ---
    # Single NNP / NN directly as NP (important: allows NNP to be subject/object)
    Production("NP",  ("NNP",)),    # NNP → NP  (single proper noun)
    Production("NP",  ("NN",)),     # NN  → NP  (single common noun)
    Production("NP",  ("NNPS",)),   # NNPS → NP
    Production("NP",  ("NNS",)),    # NNS → NP
    # PRP → NP (pronoun as subject)
    Production("NP",  ("PRP",)),
    # DT + NN → NP         ("a person")
    Production("NP",  ("DT",  "NN")),
    # DT NNP → NP        ("the Amazon")
    Production("NP",  ("DT",  "NNP")),
    # JJ NN  → NP        ("big company")
    Production("NP",  ("JJ",  "NN")),
    # JJ NNP → NP        ("great President")
    Production("NP",  ("JJ",  "NNP")),
    # NN NN  → NP        ("bank branch")
    Production("NP",  ("NN",  "NN")),
    # NNP NNP → NP       ("New York", "Edward Jones")
    Production("NP",  ("NNP", "NNP")),
    # NNP NN  → NP       ("Obama administration")
    Production("NP",  ("NNP", "NN")),
    # NP NP   → NP       (recursive, larger NPs)
    Production("NP",  ("NP",  "NP")),
    # NP PP   → NP       ("University of Technology")
    Production("NP",  ("NP",  "PP")),
    # CD NN   → NP       ("three people")
    Production("NP",  ("CD",  "NN")),
    # CD NNP  → NP       ("three Americans")
    Production("NP",  ("CD",  "NNP")),
    # PRP → NP
    Production("NP",  ("PRP", "NN")),   # he/she said
    # DT NP → NP
    Production("NP",  ("DT",  "NP")),

    # --- Prepositional Phrase ---
    # IN NP → PP         ("of Technology", "in New York")
    Production("PP",  ("IN",  "NP")),
    # IN NNP → PP        ("in London")
    Production("PP",  ("IN",  "NNP")),
    # IN NN  → PP        ("in town")
    Production("PP",  ("IN",  "NN")),

    # --- Adjective Phrase ---
    # RB JJ → ADJP       ("very big")
    Production("ADJP", ("RB", "JJ")),
    # JJ → ADJP
    Production("ADJP", ("JJ", "NN")),   # "big company" as ADJP in some contexts

    # --- Verb Phrase ---
    # VB NP  → VP        ("sees Edward")
    Production("VP",  ("VB",  "NP")),
    # VB PP  → VP        ("lives in London")
    Production("VP",  ("VB",  "PP")),
    # VB ADVP → VP       ("runs quickly")
    Production("VP",  ("VB",  "RB")),
    # VP PP  → VP        ("runs in New York quickly")
    Production("VP",  ("VP",  "PP")),
    # VP NP  → VP        ("gave the man a book")
    Production("VP",  ("VP",  "NP")),
    # VB VB  → VP        ("has been")
    Production("VP",  ("VB",  "VB")),

    # --- Sentence ---
    # NP VP → S          ("Edward runs")
    Production("S",   ("NP",  "VP")),
    # S PP  → S          ("Edward runs in London")
    Production("S",   ("S",   "PP")),
    # S NP  → S          ("Edward, CEO of…")
    Production("S",   ("S",   "NP")),
    # NP NP → S          ("Edward CEO")  (verbless sentence fragment)
    Production("S",   ("NP",  "NP")),
    # S PUNC → S         (allow trailing punctuation)
    Production("S",   ("S",   "PUNC")),
    # NP PUNC → S        (single-NP sentence fragment with punct)
    Production("S",   ("NP",  "PUNC")),
    # TIME_PHRASE and MON_PHRASE can act as VP extensions
    Production("S",   ("S",   "TIME_PHRASE")),
    Production("S",   ("S",   "MON_PHRASE")),
    Production("VP",  ("VP",  "TIME_PHRASE")),
    Production("VP",  ("VP",  "MON_PHRASE")),
    # VB + NNP directly → VP  (skip NP wrapping for simple transitive)
    Production("VP",  ("VB",  "NNP")),
]

ALL_PRODUCTIONS: list[Production] = TERMINAL_PRODUCTIONS + BINARY_PRODUCTIONS


# ---------------------------------------------------------------------------
# Pre-computed lookup tables (built once at import time)
# ---------------------------------------------------------------------------

def _build_lookup_tables(
    productions: list[Production],
) -> tuple[
    dict[tuple, list[str]],   # rhs_pair → [lhs, …]  (binary rules)
    dict[str,   list[str]],   # terminal_tag → [lhs, …]  (unary rules)
]:
    """Build reverse lookup tables needed by the CYK algorithm."""
    binary:   dict[tuple, list[str]] = {}
    terminal: dict[str,   list[str]] = {}

    for prod in productions:
        lhs, rhs = prod.lhs, prod.rhs
        if len(rhs) == 2:
            binary.setdefault(rhs, []).append(lhs)
        elif len(rhs) == 1:
            terminal.setdefault(rhs[0], []).append(lhs)

    return binary, terminal


BINARY_LOOKUP, TERMINAL_LOOKUP = _build_lookup_tables(ALL_PRODUCTIONS)
