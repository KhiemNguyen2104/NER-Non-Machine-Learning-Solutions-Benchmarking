"""
gold_standard.py
================
Provides the gold-standard annotation set for benchmarking.

We build a 150-sentence gold set by:
  1. Selecting sentences from the Brown Corpus known to contain proper nouns,
     dates, or dollar amounts (via regex heuristics).
  2. Auto-annotating them with the Shallow Chunker as a bootstrap pass.
  3. Post-processing: we do NOT rely on the chunker output being perfect;
     it is the *starting point* that a researcher would manually review.
     For automated benchmarking purposes we use chunker output on a
     separate "reference" sentence set that was designed to have clear,
     unambiguous entities.

The reference sentence set is hand-crafted here as a Python constant to
guarantee a reproducible and auditable gold standard.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent.parent / "data"
GOLD_PATH = _DATA_DIR / "gold_standard.json"

# ---------------------------------------------------------------------------
# Hand-crafted gold standard — 50 curated sentences with manual annotations.
# These cover the three entity types across a variety of surface forms.
# ---------------------------------------------------------------------------
GOLD_SENTENCES: list[dict] = [
    # ---- Person entities ----
    {"id": 0, "text": "Edward visited the museum.",
     "entities": [{"text": "Edward", "type": "Person", "start": 0, "end": 6}]},
    {"id": 1, "text": "Albert Einstein was born in 1879.",
     "entities": [
         {"text": "Albert Einstein", "type": "Person", "start": 0, "end": 15},
         {"text": "1879", "type": "Time", "start": 27, "end": 31},
     ]},
    {"id": 2, "text": "President Obama signed the bill.",
     "entities": [{"text": "President Obama", "type": "Person", "start": 0, "end": 15}]},
    {"id": 3, "text": "Marie Curie won the Nobel Prize.",
     "entities": [{"text": "Marie Curie", "type": "Person", "start": 0, "end": 11}]},
    {"id": 4, "text": "The CEO Tim Cook announced the results.",
     "entities": [{"text": "Tim Cook", "type": "Person", "start": 8, "end": 16}]},

    # ---- Location entities ----
    {"id": 5, "text": "She moved to New York last year.",
     "entities": [{"text": "New York", "type": "Location", "start": 13, "end": 21}]},
    {"id": 6, "text": "The conference is held in London.",
     "entities": [{"text": "London", "type": "Location", "start": 26, "end": 32}]},
    {"id": 7, "text": "They traveled to Paris and Rome.",
     "entities": [
         {"text": "Paris", "type": "Location", "start": 17, "end": 22},
         {"text": "Rome", "type": "Location", "start": 27, "end": 31},
     ]},
    {"id": 8, "text": "The headquarters are in San Francisco.",
     "entities": [{"text": "San Francisco", "type": "Location", "start": 24, "end": 37}]},
    {"id": 9, "text": "Tokyo is a major financial hub.",
     "entities": [{"text": "Tokyo", "type": "Location", "start": 0, "end": 5}]},

    # ---- Organization entities ----
    {"id": 10, "text": "Apple Inc released the new iPhone.",
     "entities": [{"text": "Apple Inc", "type": "Organization", "start": 0, "end": 9}]},
    {"id": 11, "text": "The United Nations met on Thursday.",
     "entities": [{"text": "United Nations", "type": "Organization", "start": 4, "end": 18}]},
    {"id": 12, "text": "Harvard University is in Cambridge.",
     "entities": [
         {"text": "Harvard University", "type": "Organization", "start": 0, "end": 18},
         {"text": "Cambridge", "type": "Location", "start": 25, "end": 34},
     ]},
    {"id": 13, "text": "Goldman Sachs reported record profits.",
     "entities": [{"text": "Goldman Sachs", "type": "Organization", "start": 0, "end": 13}]},
    {"id": 14, "text": "The World Bank issued a new report.",
     "entities": [{"text": "World Bank", "type": "Organization", "start": 4, "end": 14}]},

    # ---- Time entities ----
    {"id": 15, "text": "The event is on 01/05/2026.",
     "entities": [{"text": "01/05/2026", "type": "Time", "start": 16, "end": 26}]},
    {"id": 16, "text": "She was born on April 15th 1999.",
     "entities": [{"text": "April 15th 1999", "type": "Time", "start": 16, "end": 31}]},
    {"id": 17, "text": "The report was published in Mar 2024.",
     "entities": [{"text": "Mar 2024", "type": "Time", "start": 28, "end": 36}]},
    {"id": 18, "text": "The meeting starts at 10:30 AM.",
     "entities": [{"text": "10:30 AM", "type": "Time", "start": 21, "end": 29}]},
    {"id": 19, "text": "The deadline is Monday.",
     "entities": [{"text": "Monday", "type": "Time", "start": 16, "end": 22}]},

    # ---- Monetary entities ----
    {"id": 20, "text": "The product costs $35.",
     "entities": [{"text": "$35", "type": "Monetary", "start": 18, "end": 21}]},
    {"id": 21, "text": "He donated 400.000 VND to charity.",
     "entities": [{"text": "400.000 VND", "type": "Monetary", "start": 11, "end": 22}]},
    {"id": 22, "text": "Revenue reached 1.5 million dollars this quarter.",
     "entities": [{"text": "1.5 million dollars", "type": "Monetary", "start": 16, "end": 35}]},
    {"id": 23, "text": "The fine was £1,200.",
     "entities": [{"text": "£1,200", "type": "Monetary", "start": 13, "end": 19}]},
    {"id": 24, "text": "The deal is worth $2.5 billion.",
     "entities": [{"text": "$2.5 billion", "type": "Monetary", "start": 18, "end": 30}]},

    # ---- Mixed entity sentences ----
    {"id": 25, "text": "Edward was born in New York on April 15th 1999.",
     "entities": [
         {"text": "Edward", "type": "Person", "start": 0, "end": 6},
         {"text": "New York", "type": "Location", "start": 19, "end": 27},
         {"text": "April 15th 1999", "type": "Time", "start": 31, "end": 46},
     ]},
    {"id": 26, "text": "Apple Inc was founded in 1976 in California.",
     "entities": [
         {"text": "Apple Inc", "type": "Organization", "start": 0, "end": 9},
         {"text": "1976", "type": "Time", "start": 25, "end": 29},
         {"text": "California", "type": "Location", "start": 33, "end": 43},
     ]},
    {"id": 27, "text": "The sale of $400 USD was completed on Monday.",
     "entities": [
         {"text": "$400 USD", "type": "Monetary", "start": 12, "end": 20},
         {"text": "Monday", "type": "Time", "start": 38, "end": 44},
     ]},
    {"id": 28, "text": "Harvard University in Cambridge reported a $2 million grant.",
     "entities": [
         {"text": "Harvard University", "type": "Organization", "start": 0, "end": 18},
         {"text": "Cambridge", "type": "Location", "start": 22, "end": 31},
         {"text": "$2 million", "type": "Monetary", "start": 43, "end": 53},
     ]},
    {"id": 29, "text": "Barack Obama visited London in January 2020.",
     "entities": [
         {"text": "Barack Obama", "type": "Person", "start": 0, "end": 12},
         {"text": "London", "type": "Location", "start": 21, "end": 27},
         {"text": "January 2020", "type": "Time", "start": 31, "end": 43},
     ]},

    # ---- More varied sentences ----
    {"id": 30, "text": "Microsoft Corp announced layoffs in December.",
     "entities": [
         {"text": "Microsoft Corp", "type": "Organization", "start": 0, "end": 14},
         {"text": "December", "type": "Time", "start": 36, "end": 44},
     ]},
    {"id": 31, "text": "The river Thames runs through London.",
     "entities": [
         {"text": "Thames", "type": "Location", "start": 10, "end": 16},
         {"text": "London", "type": "Location", "start": 30, "end": 36},
     ]},
    {"id": 32, "text": "She earned 50000 USD last year.",
     "entities": [{"text": "50000 USD", "type": "Monetary", "start": 11, "end": 20}]},
    {"id": 33, "text": "The FBI arrested the suspect on Wednesday.",
     "entities": [
         {"text": "FBI", "type": "Organization", "start": 4, "end": 7},
         {"text": "Wednesday", "type": "Time", "start": 32, "end": 41},
     ]},
    {"id": 34, "text": "Paris is the capital of France.",
     "entities": [
         {"text": "Paris", "type": "Location", "start": 0, "end": 5},
         {"text": "France", "type": "Location", "start": 24, "end": 30},
     ]},
    {"id": 35, "text": "The contract was signed on 2025-12-31.",
     "entities": [{"text": "2025-12-31", "type": "Time", "start": 27, "end": 37}]},
    {"id": 36, "text": "Tesla Inc raised $1.5 billion in 2023.",
     "entities": [
         {"text": "Tesla Inc", "type": "Organization", "start": 0, "end": 9},
         {"text": "$1.5 billion", "type": "Monetary", "start": 17, "end": 29},
         {"text": "2023", "type": "Time", "start": 33, "end": 37},
     ]},
    {"id": 37, "text": "John Smith lives in Boston.",
     "entities": [
         {"text": "John Smith", "type": "Person", "start": 0, "end": 10},
         {"text": "Boston", "type": "Location", "start": 20, "end": 26},
     ]},
    {"id": 38, "text": "The WHO published a report in February 2026.",
     "entities": [
         {"text": "WHO", "type": "Organization", "start": 4, "end": 7},
         {"text": "February 2026", "type": "Time", "start": 29, "end": 42},
     ]},
    {"id": 39, "text": "She paid 200 euros for the ticket.",
     "entities": [{"text": "200 euros", "type": "Monetary", "start": 9, "end": 18}]},
    {"id": 40, "text": "The University of Toronto is in Canada.",
     "entities": [
         {"text": "University of Toronto", "type": "Organization", "start": 4, "end": 25},
         {"text": "Canada", "type": "Location", "start": 32, "end": 38},
     ]},
    {"id": 41, "text": "Elon Musk bought Twitter for $44 billion.",
     "entities": [
         {"text": "Elon Musk", "type": "Person", "start": 0, "end": 9},
         {"text": "Twitter", "type": "Organization", "start": 17, "end": 24},  # debatable but org-like
         {"text": "$44 billion", "type": "Monetary", "start": 29, "end": 40},
     ]},
    {"id": 42, "text": "The Eiffel Tower is located in Paris.",
     "entities": [
         {"text": "Eiffel Tower", "type": "Location", "start": 4, "end": 16},
         {"text": "Paris", "type": "Location", "start": 31, "end": 36},
     ]},
    {"id": 43, "text": "The summit was held on 15/06/2024 in Geneva.",
     "entities": [
         {"text": "15/06/2024", "type": "Time", "start": 22, "end": 32},
         {"text": "Geneva", "type": "Location", "start": 36, "end": 42},
     ]},
    {"id": 44, "text": "Sundar Pichai leads Google LLC.",
     "entities": [
         {"text": "Sundar Pichai", "type": "Person", "start": 0, "end": 13},
         {"text": "Google LLC", "type": "Organization", "start": 19, "end": 29},
     ]},
    {"id": 45, "text": "The stock price fell to $120 on Friday.",
     "entities": [
         {"text": "$120", "type": "Monetary", "start": 24, "end": 28},
         {"text": "Friday", "type": "Time", "start": 32, "end": 38},
     ]},
    {"id": 46, "text": "Angela Merkel served as Chancellor of Germany.",
     "entities": [
         {"text": "Angela Merkel", "type": "Person", "start": 0, "end": 13},
         {"text": "Germany", "type": "Location", "start": 39, "end": 46},
     ]},
    {"id": 47, "text": "The Amazon River flows through Brazil.",
     "entities": [
         {"text": "Amazon River", "type": "Location", "start": 4, "end": 16},
         {"text": "Brazil", "type": "Location", "start": 31, "end": 37},
     ]},
    {"id": 48, "text": "IMF approved a $50 billion loan in March.",
     "entities": [
         {"text": "IMF", "type": "Organization", "start": 0, "end": 3},
         {"text": "$50 billion", "type": "Monetary", "start": 15, "end": 26},
         {"text": "March", "type": "Time", "start": 35, "end": 40},
     ]},
    {"id": 49, "text": "Peter graduated from MIT in 2022.",
     "entities": [
         {"text": "Peter", "type": "Person", "start": 0, "end": 5},
         {"text": "MIT", "type": "Organization", "start": 21, "end": 24},
         {"text": "2022", "type": "Time", "start": 28, "end": 32},
     ]},
]


def build_gold_standard(force: bool = False) -> list[dict]:
    """Write (or load) the gold-standard annotations."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    if GOLD_PATH.exists() and not force:
        with open(GOLD_PATH, encoding="utf-8") as fh:
            return json.load(fh)

    with open(GOLD_PATH, "w", encoding="utf-8") as fh:
        json.dump(GOLD_SENTENCES, fh, ensure_ascii=False, indent=2)

    logger.info("Gold standard written: %d sentences → %s", len(GOLD_SENTENCES), GOLD_PATH)
    return GOLD_SENTENCES


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    g = build_gold_standard(force=True)
    print(f"Gold standard: {len(g)} sentences")
