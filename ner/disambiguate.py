"""
disambiguate.py
===============
Classify an "Object" entity into Person / Location / Organization / Object.

Decision cascade (in priority order):
  1. Multi-word exact match in location gazetteer → Location
  2. Multi-word exact match in person gazetteer   → Person
  3. Any token matches an org suffix              → Organization
  4. Any token matches an org keyword             → Organization
  5. Single-token exact match in location set     → Location
  6. Single-token exact match in person set       → Person
  7. Fallback                                     → Object

The gazetteers store lowercase strings; all lookups are case-insensitive.
"""

from __future__ import annotations

from ner.data.gazetteers import get_gazetteers

# Lazy-loaded sets for O(1) lookup
_location_set: frozenset[str] | None = None
_person_set:   frozenset[str] | None = None
_org_suffixes: frozenset[str] | None = None
_org_keywords: frozenset[str] | None = None


def _load() -> None:
    global _location_set, _person_set, _org_suffixes, _org_keywords  # noqa: PLW0603
    if _location_set is not None:
        return
    gaz = get_gazetteers()
    _location_set = frozenset(gaz["location"])
    _person_set   = frozenset(gaz["person"])
    _org_suffixes = frozenset(gaz["org_suffixes"])
    _org_keywords = frozenset(gaz["org_keywords"])


def disambiguate(text: str) -> str:
    """
    Return the fine-grained entity category for an Object entity.

    Parameters
    ----------
    text:
        The surface form of the entity (e.g. "New York", "Edward", "Apple Inc").

    Returns
    -------
    One of: "Person", "Location", "Organization", "Object".
    """
    _load()
    assert _location_set is not None  # mypy
    assert _person_set   is not None
    assert _org_suffixes is not None
    assert _org_keywords is not None

    lower = text.lower().strip()
    tokens_lower = lower.split()

    # --- Priority 1: Full-phrase location ---
    if lower in _location_set:
        return "Location"

    # --- Priority 2: Full-phrase person ---
    if lower in _person_set:
        return "Person"

    # --- Priority 3 & 4: Organization signals ---
    # Check org suffixes (last token or any token)
    last_tok = tokens_lower[-1] if tokens_lower else ""
    if last_tok in _org_suffixes:
        return "Organization"
    if any(tok in _org_suffixes for tok in tokens_lower):
        return "Organization"
    # Org keywords (multi-word acronyms / well-known orgs)
    if lower in _org_keywords:
        return "Organization"
    if any(kw in lower for kw in _org_keywords):
        return "Organization"

    # --- Priority 5: Token-level location match ---
    if any(tok in _location_set for tok in tokens_lower):
        return "Location"

    # --- Priority 6: Token-level person match ---
    if any(tok in _person_set for tok in tokens_lower):
        return "Person"

    # --- Priority 7: Suffix heuristics (no gazetteer match) ---
    # Org-like suffixes embedded in the string
    org_hints = {
        "university", "college", "institute", "inc", "corp", "ltd", "llc",
        "bank", "hospital", "church", "school", "company", "group",
    }
    if any(tok in org_hints for tok in tokens_lower):
        return "Organization"

    return "Object"
