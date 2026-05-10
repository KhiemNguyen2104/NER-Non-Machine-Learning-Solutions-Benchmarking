"""
gazetteers.py
=============
Download and cache named-entity gazetteers:
  - GeoNames cities500.txt  → Location names
  - US Census surname/first-name lists → Person names
  - Hand-crafted organisation suffix/keyword list → Organisation hints

Usage (standalone):
    python -m ner.data.gazetteers          # downloads & caches everything
"""

from __future__ import annotations

import json
import logging
import zipfile
from io import BytesIO
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent / "data"
_CACHE_PATH = _DATA_DIR / "gazetteers.json"

# ---------------------------------------------------------------------------
# Remote sources
# ---------------------------------------------------------------------------
GEONAMES_URL = "https://download.geonames.org/export/dump/cities500.zip"
# SSA baby names (publicly accessible, no auth required)
CENSUS_LAST_URL   = "https://raw.githubusercontent.com/smashew/NameDatabases/master/NamesDatabases/surnames/us.txt"
CENSUS_FEMALE_URL = "https://raw.githubusercontent.com/smashew/NameDatabases/master/NamesDatabases/first%20names/us.txt"
CENSUS_MALE_URL   = "https://raw.githubusercontent.com/smashew/NameDatabases/master/NamesDatabases/first%20names/all.txt"

# ---------------------------------------------------------------------------
# Hand-crafted organisation helpers
# ---------------------------------------------------------------------------
ORG_SUFFIXES: frozenset[str] = frozenset(
    {
        "inc", "corp", "ltd", "llc", "plc", "co", "company", "companies",
        "group", "holdings", "industries", "enterprises", "solutions",
        "services", "systems", "technologies", "tech", "labs", "laboratory",
        "laboratories", "institute", "institutes", "foundation", "fund",
        "bank", "trust", "insurance", "association", "society", "club",
        "committee", "commission", "council", "authority", "agency",
        "department", "ministry", "bureau", "office",
        # Academic
        "university", "college", "school", "academy", "institute",
    }
)

ORG_KEYWORDS: frozenset[str] = frozenset(
    {
        "united nations", "european union", "world bank", "imf", "who",
        "nato", "un", "eu", "fbi", "cia", "nsa", "nasa", "fda", "sec",
    }
)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_text(url: str, encoding: str = "utf-8") -> str:
    """Download a plain-text URL and return its content."""
    logger.info("Downloading %s …", url)
    resp = requests.get(url, timeout=60, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    buf = BytesIO()
    with tqdm(total=total, unit="B", unit_scale=True, desc=url.split("/")[-1]) as pbar:
        for chunk in resp.iter_content(chunk_size=65536):
            buf.write(chunk)
            pbar.update(len(chunk))
    return buf.getvalue().decode(encoding, errors="replace")


def _download_zip_member(url: str, member_name: str) -> str:
    """Download a ZIP archive and extract a single member as text."""
    logger.info("Downloading ZIP %s …", url)
    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    buf = BytesIO()
    with tqdm(total=total, unit="B", unit_scale=True, desc=url.split("/")[-1]) as pbar:
        for chunk in resp.iter_content(chunk_size=65536):
            buf.write(chunk)
            pbar.update(len(chunk))
    with zipfile.ZipFile(buf) as zf:
        with zf.open(member_name) as member:
            return member.read().decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _parse_geonames(text: str) -> set[str]:
    """Extract city and country names from GeoNames cities500.txt tab-separated dump.

    Columns (0-indexed):
      0 geonameid, 1 name, 2 asciiname, 3 alternatenames, ...
    We take column 1 (name) and column 2 (asciiname).
    """
    names: set[str] = set()
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        names.add(parts[1].strip().lower())
        names.add(parts[2].strip().lower())
    names.discard("")
    return names


def _parse_census_names(text: str) -> set[str]:
    """Extract names from a plain name-per-line file.

    Handles both:
      - Simple: one name per line
      - Census format: NAME  frequency  cumulative  rank
    """
    names: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Take only the first token (handles both formats)
        parts = line.split()
        if parts:
            name = parts[0].strip().lower()
            if name.isalpha():
                names.add(name)
    names.discard("")
    return names


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_gazetteers(force: bool = False) -> dict[str, list[str]]:
    """Download (or load from cache) all gazetteers.

    Returns
    -------
    dict with keys ``"location"``, ``"person"``, ``"org_suffixes"``, ``"org_keywords"``.
    Values are lists of lowercase strings.
    """
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    if _CACHE_PATH.exists() and not force:
        logger.info("Loading cached gazetteers from %s", _CACHE_PATH)
        with open(_CACHE_PATH, encoding="utf-8") as fh:
            return json.load(fh)

    logger.info("Building gazetteers from remote sources (first run)…")

    # --- Locations ---
    try:
        geo_text = _download_zip_member(GEONAMES_URL, "cities500.txt")
        locations = _parse_geonames(geo_text)
    except Exception as exc:
        logger.warning("GeoNames download failed (%s); using empty location set.", exc)
        locations = set()

    # --- Person names ---
    person: set[str] = set()
    for url in (CENSUS_LAST_URL, CENSUS_FEMALE_URL, CENSUS_MALE_URL):
        try:
            text = _download_text(url, encoding="latin-1")
            person |= _parse_census_names(text)
        except Exception as exc:
            logger.warning("Census download failed for %s (%s); skipping.", url, exc)

    gazetteers = {
        "location":    sorted(locations),
        "person":      sorted(person),
        "org_suffixes": sorted(ORG_SUFFIXES),
        "org_keywords": sorted(ORG_KEYWORDS),
    }

    with open(_CACHE_PATH, "w", encoding="utf-8") as fh:
        json.dump(gazetteers, fh, ensure_ascii=False)

    logger.info(
        "Gazetteers built: %d locations, %d person names → %s",
        len(locations), len(person), _CACHE_PATH,
    )
    return gazetteers


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_GAZETTEERS: dict[str, list[str]] | None = None


def get_gazetteers() -> dict[str, list[str]]:
    global _GAZETTEERS  # noqa: PLW0603
    if _GAZETTEERS is None:
        _GAZETTEERS = build_gazetteers()
    return _GAZETTEERS


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    g = build_gazetteers(force=True)
    print(f"Locations: {len(g['location'])}")
    print(f"Persons:   {len(g['person'])}")
    print(f"Org suffixes: {g['org_suffixes'][:5]}")
