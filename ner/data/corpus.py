"""
corpus.py
=========
Helpers for loading the benchmark corpus and gold-standard annotations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

_DATA_DIR = Path(__file__).parent.parent / "data"

CORPUS_PATH = _DATA_DIR / "test_corpus.json"
GOLD_PATH = _DATA_DIR / "gold_standard.json"


class Sentence(TypedDict):
    id: int
    text: str
    source: str       # e.g. "brown/news", "brown/romance", "synthetic"
    complexity: str   # "simple" | "complex" | "informal"


class GoldEntity(TypedDict):
    text: str
    type: str   # "Person" | "Location" | "Organization" | "Object" | "Time" | "Monetary"
    start: int  # character offset in sentence
    end: int


class GoldSentence(TypedDict):
    id: int
    text: str
    entities: list[GoldEntity]


def load_corpus() -> list[Sentence]:
    with open(CORPUS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def load_gold_standard() -> list[GoldSentence]:
    with open(GOLD_PATH, encoding="utf-8") as fh:
        return json.load(fh)
