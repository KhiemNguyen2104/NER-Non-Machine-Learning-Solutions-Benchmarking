"""
regex_patterns.py
=================
Compiled regular expressions for Time and Monetary entities.

Design notes
------------
* All patterns use named groups so callers can distinguish sub-formats.
* ``TIME_PATTERN`` covers:
    - ISO-style dates:  2026-05-01
    - Slash/dash dates: 01/05/2026, 5-1-26
    - Long dates:       April 15th 1999, Mar 2024, Jan. 5, 2020
    - Standalone year:  1999, 2026  (4-digit, guarded by word boundary)
    - Time-of-day:      10:30 AM, 23:59:01
    - Relative:         Monday, Tuesday … (day names)
    - Month names alone: January, February …
* ``MONETARY_PATTERN`` covers:
    - Symbol-first:   $35, £1,200.50, €400
    - Code-suffix:    400.000 VND, 1200 USD
    - Word-suffix:    35 dollars, 1.5 million dollars
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Time patterns
# ---------------------------------------------------------------------------

_MONTH_NAMES = (
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?"
)

_DAY_NAMES = (
    r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|"
    r"Mon|Tue|Wed|Thu|Fri|Sat|Sun)"
)

_ORDINAL_SUFFIX = r"(?:st|nd|rd|th)"

TIME_PATTERN: re.Pattern[str] = re.compile(
    r"(?P<iso_date>\b\d{4}-\d{1,2}-\d{1,2}\b)"
    r"|(?P<slash_date>\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b)"
    r"|(?P<long_date>"
    + _MONTH_NAMES
    + r"\s+\d{1,2}"
    + _ORDINAL_SUFFIX
    + r"?,?\s*\d{4})"
    r"|(?P<month_year>" + _MONTH_NAMES + r"\s+\d{4})"
    r"|(?P<month_day>" + _MONTH_NAMES + r"\s+\d{1,2}" + _ORDINAL_SUFFIX + r"?)"
    r"|(?P<standalone_month>\b" + _MONTH_NAMES + r"\b)"
    r"|(?P<day_name>\b" + _DAY_NAMES + r"\b)"
    r"|(?P<time_of_day>\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?)"
    r"|(?P<year>\b(?:1[0-9]{3}|20[0-2][0-9])\b)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Monetary patterns
# ---------------------------------------------------------------------------

_CURRENCY_SYMBOLS = r"(?P<symbol>[$£€¥₹₩₿])"
_CURRENCY_CODES = r"(?:USD|VND|EUR|GBP|JPY|CAD|AUD|CHF|CNY|KRW|INR|BTC)"
_CURRENCY_WORDS = r"(?:dollars?|cents?|euros?|pounds?|yen|yuan|rupees?|pesos?|francs?)"
_SCALE_WORDS = r"(?:\s*(?:billion|million|thousand|k|m|bn))?"
_NUMBER = r"\d[\d,]*(?:\.\d+)?"  # digits with optional grouped thousands and decimal

MONETARY_PATTERN: re.Pattern[str] = re.compile(
    r"(?P<symbol_first>"
    + _CURRENCY_SYMBOLS
    + r"\s*" + _NUMBER + _SCALE_WORDS + r")(?=[^%\d\w]|$)"
    r"|(?P<code_suffix>" + _NUMBER + r"\s*" + _CURRENCY_CODES + r")(?=[^%\d\w]|$)"
    r"|(?P<word_suffix>" + _NUMBER + _SCALE_WORDS + r"\s+" + _CURRENCY_WORDS + r")(?=[^%\d\w]|$)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def extract_time_entities(text: str) -> list[tuple[str, int, int]]:
    """Return list of (matched_text, start, end) for all Time matches."""
    return [(m.group(), m.start(), m.end()) for m in TIME_PATTERN.finditer(text)]


def extract_monetary_entities(text: str) -> list[tuple[str, int, int]]:
    """Return list of (matched_text, start, end) for all Monetary matches."""
    return [(m.group(), m.start(), m.end()) for m in MONETARY_PATTERN.finditer(text)]


if __name__ == "__main__":
    _test_sentences = [
        "Edward was born on April 15th 1999 in New York.",
        "The price is $35 and 400.000 VND.",
        "The meeting is scheduled for 01/05/2026 at 10:30 AM.",
        "Mar 2024 saw revenues of 1.5 million dollars.",
        "She earned £1,200.50 in January 2020.",
        "The event happens on Monday.",
        "ISO date: 2026-05-01.",
    ]
    for sent in _test_sentences:
        times = extract_time_entities(sent)
        moneys = extract_monetary_entities(sent)
        print(f"Sentence: {sent}")
        print(f"  TIME:     {times}")
        print(f"  MONETARY: {moneys}")
        print()
