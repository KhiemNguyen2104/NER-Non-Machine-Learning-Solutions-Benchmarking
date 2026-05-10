"""
experiments.py
==============
Four benchmarking experiments comparing Pure CFG vs Shallow Chunking NER.

Experiment 1 — CFG Accuracy
    Run CFG on the gold-standard corpus. For sentences it successfully parses,
    calculate Precision, Recall, F1-Score.

Experiment 2 — Chunking Accuracy
    Run Shallow Chunker on the same gold-standard corpus.
    Calculate and compare Precision, Recall, F1.

Experiment 3 — Robustness & Sentence Range
    Count parse failures. Bar chart of % corpus each system can process.

Experiment 4 — Time Complexity
    Measure runtime (ms) vs. sentence length N and corpus size.
    Plot O(N³) vs O(N) real-world performance curves.

All results are saved as CSV + matplotlib PNG plots under benchmark/results/.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_RESULTS_DIR = Path(__file__).parent.parent.parent / "benchmark" / "results"


# ---------------------------------------------------------------------------
# Lazy imports to avoid circular deps and keep startup fast
# ---------------------------------------------------------------------------

def _cfg_ner():
    from ner.cfg_ner import run_cfg_ner_safe
    return run_cfg_ner_safe


def _chunker_ner():
    from ner.chunker_ner import run_chunker_ner
    return run_chunker_ner


def _gold():
    from ner.benchmark.gold_standard import build_gold_standard
    return build_gold_standard()


def _corpus():
    from ner.benchmark.corpus_builder import build_corpus
    return build_corpus()


def _evaluate():
    from ner.benchmark.metrics import evaluate
    return evaluate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entity_list_to_dicts(entities) -> list[dict]:
    return [{"text": e.text, "type": e.entity_type} for e in entities]


def _run_cfg_on_gold(gold):
    cfg = _cfg_ner()
    results = []
    failures = 0
    for gs in gold:
        preds, success = cfg(gs["text"])
        if not success:
            failures += 1
        results.append(_entity_list_to_dicts(preds))
    return results, failures


def _run_chunker_on_gold(gold):
    chunker = _chunker_ner()
    results = []
    for gs in gold:
        preds = chunker(gs["text"])
        results.append(_entity_list_to_dicts(preds))
    return results


# ---------------------------------------------------------------------------
# Experiment 1 & 2 — Accuracy
# ---------------------------------------------------------------------------

def experiment_accuracy(save_dir: Optional[Path] = None) -> dict:
    """Run Experiments 1 & 2: accuracy comparison on gold standard."""
    save_dir = save_dir or _RESULTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    gold = _gold()
    evaluate = _evaluate()

    logger.info("Exp 1 — CFG accuracy…")
    cfg_preds, cfg_failures = _run_cfg_on_gold(gold)

    # For Exp 1: only evaluate on sentences CFG parsed
    cfg_success_gold  = [g for g, p in zip(gold, cfg_preds) if p is not None]
    cfg_success_preds = [p for p in cfg_preds]   # empty list for failures, still scored

    cfg_report = evaluate(gold, cfg_preds)
    cfg_report.print_report("Experiment 1 — CFG Accuracy")

    logger.info("Exp 2 — Chunker accuracy…")
    chunker_preds = _run_chunker_on_gold(gold)
    chunker_report = evaluate(gold, chunker_preds)
    chunker_report.print_report("Experiment 2 — Chunker Accuracy")

    summary = {
        "cfg_failures": cfg_failures,
        "cfg_total": len(gold),
        "cfg_parse_rate": round((len(gold) - cfg_failures) / len(gold), 4),
        "cfg_metrics": cfg_report.to_dict(),
        "chunker_metrics": chunker_report.to_dict(),
    }

    out = save_dir / "accuracy_results.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Accuracy results saved to %s", out)

    # CSV
    csv_out = save_dir / "accuracy_comparison.csv"
    with open(csv_out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "type", "precision", "recall", "f1", "tp", "fp", "fn"])
        for etype, m in sorted(cfg_report.per_type.items()):
            w.writerow(["CFG", etype] + list(m.to_dict().values()))
        m = cfg_report.overall
        w.writerow(["CFG", "OVERALL"] + list(m.to_dict().values()))
        for etype, m in sorted(chunker_report.per_type.items()):
            w.writerow(["Chunker", etype] + list(m.to_dict().values()))
        m = chunker_report.overall
        w.writerow(["Chunker", "OVERALL"] + list(m.to_dict().values()))

    return summary


# ---------------------------------------------------------------------------
# Experiment 3 — Robustness
# ---------------------------------------------------------------------------

def experiment_robustness(save_dir: Optional[Path] = None) -> dict:
    """Experiment 3: count parse failures; plot % corpus processed."""
    save_dir = save_dir or _RESULTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    corpus = _corpus()
    cfg = _cfg_ner()
    chunker = _chunker_ner()

    cfg_fail = 0
    chunker_fail = 0   # chunker never fails, but we check for empty output

    logger.info("Exp 3 — Robustness on %d sentences…", len(corpus))
    for item in corpus:
        _, success = cfg(item["text"])
        if not success:
            cfg_fail += 1
        preds = chunker(item["text"])
        if not preds:
            chunker_fail += 1

    total = len(corpus)
    summary = {
        "total_sentences": total,
        "cfg_failures": cfg_fail,
        "cfg_success_pct": round((total - cfg_fail) / total * 100, 2),
        "chunker_no_output": chunker_fail,
        "chunker_success_pct": round((total - chunker_fail) / total * 100, 2),
    }

    print(f"\nExp 3 — Robustness")
    print(f"  CFG:     parsed {total - cfg_fail}/{total} ({summary['cfg_success_pct']}%)")
    print(f"  Chunker: output {total - chunker_fail}/{total} ({summary['chunker_success_pct']}%)")

    out = save_dir / "robustness_results.json"
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)

    _plot_robustness(summary, save_dir)
    return summary


def _plot_robustness(summary: dict, save_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        systems = ["CFG", "Chunker"]
        success = [summary["cfg_success_pct"], summary["chunker_success_pct"]]
        failure = [100 - s for s in success]

        fig, ax = plt.subplots(figsize=(7, 5))
        x = range(len(systems))
        bars_s = ax.bar(x, success, 0.5, label="Successfully Processed", color=["#2196F3", "#4CAF50"])
        bars_f = ax.bar(x, failure, 0.5, bottom=success, label="Failed / No Output",
                        color=["#F44336", "#FF9800"])

        for bar, val in zip(bars_s, success):
            ax.text(bar.get_x() + bar.get_width() / 2, val / 2,
                    f"{val:.1f}%", ha="center", va="center", color="white", fontweight="bold")
        for bar, bot, val in zip(bars_f, success, failure):
            if val > 2:
                ax.text(bar.get_x() + bar.get_width() / 2, bot + val / 2,
                        f"{val:.1f}%", ha="center", va="center", color="white", fontweight="bold")

        ax.set_xticks(list(x))
        ax.set_xticklabels(systems, fontsize=13)
        ax.set_ylabel("% of Corpus", fontsize=12)
        ax.set_title(f"Experiment 3 — Robustness\n(N={summary['total_sentences']} sentences)", fontsize=13)
        ax.legend()
        ax.set_ylim(0, 110)
        plt.tight_layout()
        plt.savefig(save_dir / "exp3_robustness.png", dpi=150)
        plt.close()
        logger.info("Robustness plot saved.")
    except Exception as e:
        logger.warning("Could not generate robustness plot: %s", e)


# ---------------------------------------------------------------------------
# Experiment 4 — Time Complexity
# ---------------------------------------------------------------------------

def experiment_time_complexity(
    max_n: int = 20,
    repeats: int = 5,
    save_dir: Optional[Path] = None,
) -> dict:
    """
    Experiment 4: measure runtime (ms) as sentence length N increases.

    Synthetic sentences of length N are constructed by repeating
    "The great leader Edward visited New York." tokens up to N words.
    """
    save_dir = save_dir or _RESULTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    cfg = _cfg_ner()
    chunker = _chunker_ner()

    base_tokens = (
        "The great leader Edward visited beautiful New York last January "
        "and signed a $50 million contract with Apple Inc corporation"
    ).split()

    lengths = list(range(2, max_n + 1, 2))
    cfg_times:     list[float] = []
    chunker_times: list[float] = []

    logger.info("Exp 4 — Time complexity (N = %s, %d repeats each)…", lengths, repeats)

    for n in lengths:
        tokens = (base_tokens * ((n // len(base_tokens)) + 1))[:n]
        sentence = " ".join(tokens)

        # CFG
        t_cfg = 0.0
        for _ in range(repeats):
            t0 = time.perf_counter()
            cfg(sentence)
            t_cfg += (time.perf_counter() - t0) * 1000
        cfg_times.append(t_cfg / repeats)

        # Chunker
        t_chk = 0.0
        for _ in range(repeats):
            t0 = time.perf_counter()
            chunker(sentence)
            t_chk += (time.perf_counter() - t0) * 1000
        chunker_times.append(t_chk / repeats)

    print("\nExp 4 — Time Complexity (avg ms per sentence)")
    print(f"{'N':>5} {'CFG (ms)':>12} {'Chunker (ms)':>14}")
    for n, ct, ckt in zip(lengths, cfg_times, chunker_times):
        print(f"{n:>5} {ct:>12.3f} {ckt:>14.3f}")

    summary = {"lengths": lengths, "cfg_ms": cfg_times, "chunker_ms": chunker_times}
    out = save_dir / "time_complexity.json"
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)

    # CSV
    csv_out = save_dir / "time_complexity.csv"
    with open(csv_out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["N", "cfg_ms", "chunker_ms"])
        for row in zip(lengths, cfg_times, chunker_times):
            w.writerow(row)

    _plot_time(lengths, cfg_times, chunker_times, save_dir)
    return summary


def _plot_time(
    lengths: list[int],
    cfg_times: list[float],
    chunker_times: list[float],
    save_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Left: linear scale
        axes[0].plot(lengths, cfg_times,     "o-", color="#F44336", label="CFG  O(N³)", linewidth=2)
        axes[0].plot(lengths, chunker_times, "s-", color="#4CAF50", label="Chunker O(N)", linewidth=2)
        axes[0].set_xlabel("Sentence Length (N tokens)", fontsize=12)
        axes[0].set_ylabel("Avg Runtime (ms)", fontsize=12)
        axes[0].set_title("Runtime vs. Sentence Length\n(linear scale)", fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Right: log scale
        axes[1].semilogy(lengths, cfg_times,     "o-", color="#F44336", label="CFG  O(N³)", linewidth=2)
        axes[1].semilogy(lengths, chunker_times, "s-", color="#4CAF50", label="Chunker O(N)", linewidth=2)
        axes[1].set_xlabel("Sentence Length (N tokens)", fontsize=12)
        axes[1].set_ylabel("Avg Runtime (ms, log scale)", fontsize=12)
        axes[1].set_title("Runtime vs. Sentence Length\n(log scale)", fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, which="both")

        plt.suptitle("Experiment 4 — Time Complexity: CFG O(N³) vs Chunker O(N)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_dir / "exp4_time_complexity.png", dpi=150)
        plt.close()
        logger.info("Time complexity plot saved.")
    except Exception as e:
        logger.warning("Could not generate time plot: %s", e)


# ---------------------------------------------------------------------------
# Run all experiments
# ---------------------------------------------------------------------------

def run_all(save_dir: Optional[Path] = None) -> None:
    """Run all four experiments and save results."""
    save_dir = save_dir or _RESULTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting all benchmark experiments…")
    acc   = experiment_accuracy(save_dir)
    robust = experiment_robustness(save_dir)
    time_c = experiment_time_complexity(save_dir=save_dir)

    master = {"accuracy": acc, "robustness": robust, "time_complexity": time_c}
    with open(save_dir / "all_results.json", "w") as fh:
        json.dump(master, fh, indent=2)
    logger.info("All experiments complete. Results in %s", save_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_all()
