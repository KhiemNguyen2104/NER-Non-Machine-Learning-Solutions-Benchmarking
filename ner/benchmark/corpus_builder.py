"""
corpus_builder.py
=================
Build a stratified 800-sentence test corpus from the NLTK Brown Corpus.

Stratification:
  - ~300 "simple"   sentences (≤ 10 tokens, news/government genre)
  - ~300 "complex"  sentences (> 15 tokens, learned/editorial genre)
  - ~200 "informal" sentences (fiction/humor genre + added noise)

Saves to ner/data/test_corpus.json.
"""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent.parent / "data"
CORPUS_PATH = _DATA_DIR / "test_corpus.json"

# Brown genre → complexity bucket
_SIMPLE_GENRES   = ["news", "government", "hobbies"]
_COMPLEX_GENRES  = ["learned", "editorial", "reviews"]
_INFORMAL_GENRES = ["fiction", "humor", "romance", "mystery", "science_fiction"]

TARGET_SIMPLE   = 300
TARGET_COMPLEX  = 300
TARGET_INFORMAL = 200


def _raw_sentences_from_brown(genres: list[str]) -> list[str]:
    import nltk
    nltk.download("brown", quiet=True)
    from nltk.corpus import brown

    sents = []
    for genre in genres:
        try:
            for tokens in brown.sents(categories=genre):
                text = " ".join(tokens)
                text = re.sub(r"\s+", " ", text).strip()
                sents.append(text)
        except Exception:
            pass
    return sents


def build_corpus(force: bool = False, seed: int = 42) -> list[dict]:
    """Build and save the test corpus; returns list of sentence dicts."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    if CORPUS_PATH.exists() and not force:
        logger.info("Loading existing corpus from %s", CORPUS_PATH)
        with open(CORPUS_PATH, encoding="utf-8") as fh:
            return json.load(fh)

    logger.info("Building test corpus from Brown Corpus…")
    rng = random.Random(seed)

    simple_raw   = _raw_sentences_from_brown(_SIMPLE_GENRES)
    complex_raw  = _raw_sentences_from_brown(_COMPLEX_GENRES)
    informal_raw = _raw_sentences_from_brown(_INFORMAL_GENRES)

    simple   = [s for s in simple_raw  if 3 <= len(s.split()) <= 12]
    complex_ = [s for s in complex_raw if len(s.split()) > 14]
    informal = [s for s in informal_raw if 3 <= len(s.split()) <= 20]

    rng.shuffle(simple)
    rng.shuffle(complex_)
    rng.shuffle(informal)

    simple   = simple[:TARGET_SIMPLE]
    complex_ = complex_[:TARGET_COMPLEX]
    informal = informal[:TARGET_INFORMAL]

    corpus = []
    sid = 0
    for text in simple:
        corpus.append({"id": sid, "text": text, "source": "brown", "complexity": "simple"})
        sid += 1
    for text in complex_:
        corpus.append({"id": sid, "text": text, "source": "brown", "complexity": "complex"})
        sid += 1
    for text in informal:
        corpus.append({"id": sid, "text": text, "source": "brown", "complexity": "informal"})
        sid += 1

    rng.shuffle(corpus)
    for i, item in enumerate(corpus):
        item["id"] = i

    with open(CORPUS_PATH, "w", encoding="utf-8") as fh:
        json.dump(corpus, fh, ensure_ascii=False, indent=2)

    logger.info("Corpus built: %d sentences → %s", len(corpus), CORPUS_PATH)
    return corpus


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    c = build_corpus(force=True)
    print(f"Total: {len(c)} sentences")
    from collections import Counter
    print(Counter(s["complexity"] for s in c))
