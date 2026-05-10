#!/usr/bin/env python3
"""
main.py
=======
CLI entry point for the Rule-Based NER system.

Commands
--------
    python main.py setup              Download data & build caches
    python main.py setup --force      Force re-download & rebuild

    python main.py run-chunker        Interactive Shallow Chunker session
    python main.py run-cfg            Interactive CFG NER session

    python main.py demo               Run both pipelines on built-in examples
    python main.py benchmark          Run all 4 benchmark experiments
    python main.py benchmark --quick  Only accuracy experiments (no full corpus)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path when run directly
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Interactive session helpers
# ---------------------------------------------------------------------------

def _interactive_chunker() -> None:
    from ner.chunker_ner import run_chunker_ner
    print("\n── Shallow Chunker NER ── (type 'quit' to exit)\n")
    while True:
        try:
            sentence = input("Enter sentence: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if sentence.lower() in {"quit", "exit", "q"}:
            break
        if not sentence:
            continue
        entities = run_chunker_ner(sentence)
        if entities:
            print("  Entities found:")
            for e in entities:
                print(f"    [{e.entity_type}]  {e.text!r}")
        else:
            print("  No entities found.")
        print()


def _interactive_cfg() -> None:
    from ner.cfg_ner import run_cfg_ner, ParseError
    print("\n── Pure CFG NER ── (type 'quit' to exit)\n")
    while True:
        try:
            sentence = input("Enter sentence: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if sentence.lower() in {"quit", "exit", "q"}:
            break
        if not sentence:
            continue
        try:
            entities = run_cfg_ner(sentence)
            if entities:
                print("  Entities found:")
                for e in entities:
                    print(f"    [{e.entity_type}]  {e.text!r}")
            else:
                print("  No entities found in parse tree.")
        except ParseError as exc:
            print(f"  ✗ Parse failed: {exc}")
        print()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

DEMO_SENTENCES = [
    "Edward was born in New York on April 15th 1999.",
    "Apple Inc released the new iPhone for $999.",
    "The United Nations met in Geneva on 15/06/2024.",
    "Barack Obama donated $10 million to Harvard University.",
    "Mar 2024 revenues exceeded 1.5 billion dollars.",
    "She moved to Tokyo and works for Sony Corp.",
    "The FBI arrested the suspect on Monday.",
    "He earned 50000 VND per day last year.",
]


def _run_demo() -> None:
    from ner.cfg_ner import run_cfg_ner_safe
    from ner.chunker_ner import run_chunker_ner

    print("\n" + "=" * 65)
    print("  NER Demo — Comparing CFG vs Shallow Chunker")
    print("=" * 65)

    for sent in DEMO_SENTENCES:
        print(f"\nSentence: {sent}")

        # Chunker (always works)
        chk_ents = run_chunker_ner(sent)
        print("  [Chunker]")
        if chk_ents:
            for e in chk_ents:
                print(f"    [{e.entity_type}] {e.text!r}")
        else:
            print("    (no entities)")

        # CFG
        cfg_ents, success = run_cfg_ner_safe(sent)
        print("  [CFG]")
        if not success:
            print("    ✗ Parse failed")
        elif cfg_ents:
            for e in cfg_ents:
                print(f"    [{e.entity_type}] {e.text!r}")
        else:
            print("    (no entities in parse tree)")

    print("\n" + "=" * 65)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def _run_benchmark(quick: bool = False) -> None:
    from ner.benchmark.experiments import experiment_accuracy, experiment_robustness, experiment_time_complexity

    results_dir = _ROOT / "benchmark" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    experiment_accuracy(save_dir=results_dir)
    if not quick:
        experiment_robustness(save_dir=results_dir)
        experiment_time_complexity(save_dir=results_dir)

    print(f"\nResults saved to: {results_dir}/")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rule-Based NER: Pure CFG vs Shallow Chunking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # setup
    sp = sub.add_parser("setup", help="Download data and build caches")
    sp.add_argument("--force", action="store_true", help="Force re-download/rebuild")

    # run-chunker
    sub.add_parser("run-chunker", help="Interactive Shallow Chunker NER session")

    # run-cfg
    sub.add_parser("run-cfg", help="Interactive Pure CFG NER session")

    # demo
    sub.add_parser("demo", help="Run both pipelines on built-in example sentences")

    # benchmark
    bp = sub.add_parser("benchmark", help="Run all 4 benchmark experiments")
    bp.add_argument("--quick", action="store_true",
                    help="Only run accuracy experiments (skip robustness & time)")

    args = parser.parse_args()

    if args.command == "setup":
        from ner.setup import run_setup
        run_setup(force=args.force)

    elif args.command == "run-chunker":
        _interactive_chunker()

    elif args.command == "run-cfg":
        _interactive_cfg()

    elif args.command == "demo":
        _run_demo()

    elif args.command == "benchmark":
        _run_benchmark(quick=args.quick)


if __name__ == "__main__":
    main()
