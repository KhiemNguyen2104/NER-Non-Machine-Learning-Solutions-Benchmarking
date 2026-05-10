"""
chunker.py
==========
FSM-style regex chunker over POS-tagged token sequences.

Pipeline
--------
1. Apply TIME_PATTERN and MONETARY_PATTERN to the raw sentence text.
   Collect (text, start, end, "Time"/"Monetary") spans.
   Remove/mask those spans to prevent double-tagging.

2. Tokenise the remaining (masked) text.
   Tag tokens using the O(1) UnigramTagger.

3. Iterate through the POS-tagged token list with the FSM rules in
   priority order (as defined in chunk_grammar.FSM_RULES).
   Emit (text, "Object") for each matching chunk.

4. Return a merged, de-overlapped list of NERResult namedtuples.

The FSM engine is O(N) per rule (linear scan with backtracking only on
rule failure at the current position, not globally), making the full pass
O(N × |rules|) = O(N) since |rules| is a constant.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from ner.patterns.regex_patterns import TIME_PATTERN, MONETARY_PATTERN
from ner.patterns.chunk_grammar import FSM_RULES
from ner.chunker.pos_tagger import get_tagger


class NERSpan(NamedTuple):
    text: str
    entity_type: str   # "Time" | "Monetary" | "Object" (to be disambiguated)
    start: int         # character offset in original sentence
    end: int


# ---------------------------------------------------------------------------
# Step 1 helper – regex extraction with span tracking
# ---------------------------------------------------------------------------

def _extract_regex_spans(sentence: str) -> list[NERSpan]:
    """Return all Time and Monetary spans, non-overlapping (first wins)."""
    raw: list[tuple[int, int, str]] = []
    for m in TIME_PATTERN.finditer(sentence):
        raw.append((m.start(), m.end(), "Time"))
    for m in MONETARY_PATTERN.finditer(sentence):
        raw.append((m.start(), m.end(), "Monetary"))
    raw.sort(key=lambda x: x[0])

    spans: list[NERSpan] = []
    last_end = -1
    for s, e, etype in raw:
        if s >= last_end:
            spans.append(NERSpan(sentence[s:e], etype, s, e))
            last_end = e
    return spans


def _mask_spans(sentence: str, spans: list[NERSpan]) -> str:
    """Replace extracted spans with whitespace to prevent re-matching."""
    chars = list(sentence)
    for span in spans:
        for i in range(span.start, span.end):
            chars[i] = " "
    return "".join(chars)


# ---------------------------------------------------------------------------
# Step 3 helper – FSM chunker
# ---------------------------------------------------------------------------

_TOKENISE_RE = re.compile(r"[^\s]+")


def _tokenise_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """Return (token, start, end) triples from whitespace-split tokenisation."""
    return [(m.group(), m.start(), m.end()) for m in _TOKENISE_RE.finditer(text)]


def _strip_punctuation(token: str) -> str:
    return token.strip(".,;:!?\"'()[]{}—–-")


def _apply_fsm(
    tagged: list[tuple[str, str, int, int]],  # (token, tag, char_start, char_end)
) -> list[tuple[int, int]]:
    """
    Apply FSM_RULES left-to-right over *tagged* token list.
    Returns list of (token_start_idx, token_end_idx) inclusive index pairs
    for each matched chunk (Object NP candidates only).
    """
    n = len(tagged)
    chunks: list[tuple[int, int]] = []
    i = 0

    while i < n:
        matched = False
        for rule in FSM_RULES:
            sequence = rule["sequence"]
            pos = i
            seg_matches: list[tuple[int, int]] = []   # (start_idx, end_idx) per element

            valid = True
            for element in sequence:
                accepted_tags = element["tags"]
                repeat = element["repeat"]

                if repeat == "1":
                    if pos < n and tagged[pos][1] in accepted_tags:
                        seg_matches.append((pos, pos))
                        pos += 1
                    else:
                        valid = False
                        break

                elif repeat == "+":
                    start_pos = pos
                    while pos < n and tagged[pos][1] in accepted_tags:
                        pos += 1
                    if pos > start_pos:
                        seg_matches.append((start_pos, pos - 1))
                    else:
                        valid = False
                        break

                elif repeat == "*":
                    start_pos = pos
                    while pos < n and tagged[pos][1] in accepted_tags:
                        pos += 1
                    seg_matches.append((start_pos, pos - 1))  # may be empty

                elif repeat == "?":
                    if pos < n and tagged[pos][1] in accepted_tags:
                        seg_matches.append((pos, pos))
                        pos += 1
                    else:
                        seg_matches.append((pos, pos - 1))   # empty match

            if valid and pos > i:
                chunk_start = i
                chunk_end = pos - 1
                chunks.append((chunk_start, chunk_end))
                i = pos
                matched = True
                break

        if not matched:
            i += 1

    return chunks


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def chunk(sentence: str) -> list[NERSpan]:
    """
    Run the full shallow-chunking NER pipeline on *sentence*.

    Returns a list of NERSpan named tuples with entity_type ∈
    {"Time", "Monetary", "Object"}.  Object entities need disambiguation.
    """
    tagger = get_tagger()

    # Step 1 – regex extraction
    regex_spans = _extract_regex_spans(sentence)

    # Step 2 – mask and tokenise remaining text
    masked = _mask_spans(sentence, regex_spans)
    raw_tokens = _tokenise_with_offsets(masked)

    # Clean and tag tokens (strip trailing punctuation for tagging)
    tagged_tokens: list[tuple[str, str, int, int]] = []
    sentence_start_done = False  # first content token = position 0 (sentence start)
    for idx, (tok, s, e) in enumerate(raw_tokens):
        clean = _strip_punctuation(tok)
        if not clean:
            continue
        # Pass token index so the tagger knows whether this is sentence-start
        tag = tagger.tag_token(clean, position=idx)
        # Additional heuristic: if token is capitalised and got tagged NOUN
        # (Brown sees it as common noun) but it's not the first token, it's
        # likely a proper noun in context.
        if clean[0].isupper() and idx > 0 and tag == "NOUN":
            tag = "PROPN"
        tagged_tokens.append((clean, tag, s, e))


    # Step 3 – FSM chunking
    chunk_indices = _apply_fsm(tagged_tokens)

    object_spans: list[NERSpan] = []
    for start_idx, end_idx in chunk_indices:
        chunk_tokens = tagged_tokens[start_idx: end_idx + 1]
        # Only emit as Object if at least one token is PROPN
        # (eliminates common-noun phrases like 'the new product')
        if not any(t[1] == "PROPN" for t in chunk_tokens):
            continue
        text = " ".join(t[0] for t in chunk_tokens)
        char_start = chunk_tokens[0][2]
        char_end   = chunk_tokens[-1][3]
        object_spans.append(NERSpan(text, "Object", char_start, char_end))

    # Step 4 – merge and sort all spans by character offset
    all_spans = regex_spans + object_spans
    all_spans.sort(key=lambda s: s.start)
    return all_spans
