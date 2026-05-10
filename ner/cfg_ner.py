"""
cfg_ner.py
==========
Top-level CFG NER pipeline.

Implements the pseudo-code from the task description:
    CFG_NER_Extractor(sentence, grammar, lexicon)

Returns a list of Entity namedtuples or raises ParseError on failure.
"""

from __future__ import annotations

from typing import NamedTuple

from ner.cfg.lexicon import get_lexicon
from ner.cfg.cyk_parser import parse, extract_entities
from ner.disambiguate import disambiguate


class Entity(NamedTuple):
    text: str
    entity_type: str   # "Person" | "Location" | "Organization" | "Object" | "Time" | "Monetary"


class ParseError(Exception):
    """Raised when the CFG cannot parse the input sentence."""


def run_cfg_ner(sentence: str) -> list[Entity]:
    """
    Extract named entities from *sentence* using the Pure CFG pipeline.

    Parameters
    ----------
    sentence:
        Raw English sentence text.

    Returns
    -------
    list[Entity]
        Named entities found in the sentence.

    Raises
    ------
    ParseError
        If the CYK parser cannot build a complete parse tree (S → …).
    """
    lexicon = get_lexicon()

    # Step 1 – Lexical mapping (injects TIME_TERMINAL / MON_TERMINAL)
    tagged = lexicon.tag_sentence(sentence)

    if not tagged:
        return []

    # Step 2 – Build syntax tree via CYK
    tree = parse(tagged)

    # Step 3 – Handle parse failure
    if tree is None:
        raise ParseError(f"CFG could not parse: {sentence!r}")

    # Step 4 – Tree traversal for extraction
    raw_entities = extract_entities(tree)

    # Step 5 – Disambiguate Object → Person / Location / Organization
    entities: list[Entity] = []
    for text, raw_type in raw_entities:
        if raw_type == "Object":
            fine_type = disambiguate(text)
        else:
            fine_type = raw_type
        entities.append(Entity(text=text, entity_type=fine_type))

    return entities


def run_cfg_ner_safe(sentence: str) -> tuple[list[Entity], bool]:
    """
    Like ``run_cfg_ner`` but catches ``ParseError`` and returns a success flag.

    Returns
    -------
    (entities, success)
        If success is False, entities is an empty list.
    """
    try:
        return run_cfg_ner(sentence), True
    except ParseError:
        return [], False
