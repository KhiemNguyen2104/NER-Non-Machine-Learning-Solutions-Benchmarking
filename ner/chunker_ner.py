"""
chunker_ner.py
==============
Top-level Shallow Chunking NER pipeline.

Implements the pseudo-code from the task description:
    Chunker_NER_Extractor(sentence, pos_dictionary, chunk_grammar)
"""

from __future__ import annotations

from typing import NamedTuple

from ner.chunker.chunker import chunk, NERSpan
from ner.disambiguate import disambiguate


class Entity(NamedTuple):
    text: str
    entity_type: str   # "Person" | "Location" | "Organization" | "Object" | "Time" | "Monetary"


def run_chunker_ner(sentence: str) -> list[Entity]:
    """
    Extract named entities from *sentence* using Shallow Chunking.

    Parameters
    ----------
    sentence:
        Raw English sentence text.

    Returns
    -------
    list[Entity]
        Named entities, always returns (never raises on parse failure).
    """
    spans: list[NERSpan] = chunk(sentence)

    entities: list[Entity] = []
    for span in spans:
        if span.entity_type == "Object":
            fine_type = disambiguate(span.text)
        else:
            fine_type = span.entity_type
        entities.append(Entity(text=span.text, entity_type=fine_type))

    return entities
