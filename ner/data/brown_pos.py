"""
brown_pos.py
============
Build a O(1) unigram POS dictionary from the NLTK Brown Corpus.

Usage (standalone):
    python -m ner.data.brown_pos          # writes data/brown_unigram.json
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent / "data"
_CACHE_PATH = _DATA_DIR / "brown_unigram.json"

# Universal → simplified tag mapping used internally throughout the project.
# Maps NLTK Universal tagset names to our canonical tag strings.
UNIVERSAL_TAG_MAP: dict[str, str] = {
    "NOUN": "NOUN",
    "VERB": "VERB",
    "ADJ":  "ADJ",
    "ADV":  "ADV",
    "PRON": "PRON",
    "DET":  "DET",
    "ADP":  "ADP",
    "NUM":  "NUM",
    "CONJ": "CONJ",
    "PRT":  "PRT",
    ".":    ".",
    "X":    "X",
}


def build_brown_unigram_dict(force: bool = False) -> dict[str, str]:
    """Build (or load from cache) the Brown Corpus unigram POS dictionary.

    For each unique lowercase word, we record the single most-frequent
    Universal POS tag seen across the entire Brown Corpus.

    Returns
    -------
    dict[str, str]
        Mapping of ``word (lower-cased) → most-frequent Universal POS tag``.
    """
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    if _CACHE_PATH.exists() and not force:
        logger.info("Loading cached Brown unigram dict from %s", _CACHE_PATH)
        with open(_CACHE_PATH, encoding="utf-8") as fh:
            return json.load(fh)

    logger.info("Building Brown unigram dict (first run, may take a moment)…")
    import nltk  # noqa: PLC0415

    # Ensure the corpus is available
    nltk.download("brown", quiet=True)
    nltk.download("universal_tagset", quiet=True)

    from nltk.corpus import brown  # noqa: PLC0415

    tag_counts: dict[str, Counter] = defaultdict(Counter)
    for word, tag in brown.tagged_words(tagset="universal"):
        tag_counts[word.lower()][tag] += 1

    unigram: dict[str, str] = {
        word: counter.most_common(1)[0][0]
        for word, counter in tag_counts.items()
    }

    with open(_CACHE_PATH, "w", encoding="utf-8") as fh:
        json.dump(unigram, fh, ensure_ascii=False)

    logger.info("Brown unigram dict built: %d words → %s", len(unigram), _CACHE_PATH)
    return unigram


# ---------------------------------------------------------------------------
# Convenience singleton – loaded lazily.
# ---------------------------------------------------------------------------
_UNIGRAM_DICT: dict[str, str] | None = None


def get_unigram_dict() -> dict[str, str]:
    """Return the singleton Brown unigram dict, building/loading it if needed."""
    global _UNIGRAM_DICT  # noqa: PLW0603
    if _UNIGRAM_DICT is None:
        _UNIGRAM_DICT = build_brown_unigram_dict()
    return _UNIGRAM_DICT


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    d = build_brown_unigram_dict(force=True)
    print(f"Dictionary size: {len(d)} entries")
    sample = {k: d[k] for k in list(d)[:10]}
    print("Sample entries:", sample)
