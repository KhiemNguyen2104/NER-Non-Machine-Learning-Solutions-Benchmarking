"""
setup.py  (inside ner package)
===============================
One-shot data bootstrap: downloads corpora, builds lexicons, writes caches.
Called by ``python main.py setup``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def run_setup(force: bool = False) -> None:
    """Download all remote data and build all local caches."""

    print("=" * 55)
    print("  NER Setup — downloading corpora & building caches")
    print("=" * 55)

    # 1. Brown Corpus unigram POS dict
    print("\n[1/5] Building Brown Corpus unigram POS dictionary…")
    from ner.data.brown_pos import build_brown_unigram_dict
    d = build_brown_unigram_dict(force=force)
    print(f"      Done — {len(d)} word entries")

    # 2. Gazetteers
    print("\n[2/5] Downloading gazetteers (GeoNames + US Census)…")
    from ner.data.gazetteers import build_gazetteers
    g = build_gazetteers(force=force)
    print(f"      Done — {len(g['location'])} locations, {len(g['person'])} persons")

    # 3. CFG Lexicon
    print("\n[3/5] Building CFG lexicon…")
    from ner.cfg.lexicon import Lexicon
    lex = Lexicon().build(force=force)
    print(f"      Done — {len(lex._word_to_tag)} lexicon entries")

    # 4. Test corpus
    print("\n[4/5] Building test corpus from Brown Corpus…")
    from ner.benchmark.corpus_builder import build_corpus
    c = build_corpus(force=force)
    print(f"      Done — {len(c)} sentences")

    # 5. Gold standard
    print("\n[5/5] Writing gold-standard annotations…")
    from ner.benchmark.gold_standard import build_gold_standard
    gs = build_gold_standard(force=force)
    print(f"      Done — {len(gs)} annotated sentences")

    print("\n✓ Setup complete. Run:")
    print("    python main.py run-chunker")
    print("    python main.py run-cfg")
    print("    python main.py benchmark")
