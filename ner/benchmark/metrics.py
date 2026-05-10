"""
metrics.py
==========
Span-level NER evaluation metrics: Precision, Recall, F1-Score.

Matching strategy: exact span match (text + entity_type must both match).
We also support type-agnostic span matching (text only) for looser analysis.

Entity types evaluated:
  - Person, Location, Organization, Object, Time, Monetary
  - "Object" macro-category (aggregates Person + Location + Organization + Object)
  - Overall (all types)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TypedDict


class EntityAnnotation(TypedDict):
    text: str
    type: str
    start: int
    end: int


@dataclass
class MetricResult:
    true_positives:  int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def __add__(self, other: "MetricResult") -> "MetricResult":
        return MetricResult(
            self.true_positives  + other.true_positives,
            self.false_positives + other.false_positives,
            self.false_negatives + other.false_negatives,
        )

    def to_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall":    round(self.recall, 4),
            "f1":        round(self.f1, 4),
            "tp": self.true_positives,
            "fp": self.false_positives,
            "fn": self.false_negatives,
        }


@dataclass
class EvaluationReport:
    per_type: dict[str, MetricResult] = field(default_factory=dict)
    overall:  MetricResult = field(default_factory=MetricResult)

    def to_dict(self) -> dict:
        return {
            "per_type": {k: v.to_dict() for k, v in self.per_type.items()},
            "overall":  self.overall.to_dict(),
        }

    def print_report(self, title: str = "Evaluation Report") -> None:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
        header = f"{'Type':<20} {'P':>8} {'R':>8} {'F1':>8} {'TP':>6} {'FP':>6} {'FN':>6}"
        print(header)
        print("-" * 60)
        for etype, m in sorted(self.per_type.items()):
            print(f"{etype:<20} {m.precision:>8.3f} {m.recall:>8.3f} {m.f1:>8.3f} "
                  f"{m.true_positives:>6} {m.false_positives:>6} {m.false_negatives:>6}")
        print("-" * 60)
        m = self.overall
        print(f"{'OVERALL':<20} {m.precision:>8.3f} {m.recall:>8.3f} {m.f1:>8.3f} "
              f"{m.true_positives:>6} {m.false_positives:>6} {m.false_negatives:>6}")
        print("=" * 60)


def _normalize(text: str) -> str:
    """Lowercase and strip for fuzzy matching."""
    return text.lower().strip()


def evaluate(
    gold_sentences: list[dict],
    predicted: list[list[dict]],  # same length as gold_sentences
    exact_type: bool = True,
) -> EvaluationReport:
    """
    Compute span-level P/R/F1 over a list of sentences.

    Parameters
    ----------
    gold_sentences:
        List of gold-standard sentence dicts with ``entities`` key.
    predicted:
        Parallel list of predicted entity lists. Each element is a list of
        dicts with at least ``text`` and ``type`` keys.
    exact_type:
        If True, both text AND type must match. If False, only text.

    Returns
    -------
    EvaluationReport
    """
    per_type: dict[str, MetricResult] = defaultdict(MetricResult)
    overall = MetricResult()

    for gold_sent, pred_ents in zip(gold_sentences, predicted):
        gold_ents: list[EntityAnnotation] = gold_sent.get("entities", [])

        if exact_type:
            gold_set = {(_normalize(e["text"]), e["type"]) for e in gold_ents}
            pred_set = {(_normalize(e["text"]), e["type"]) for e in pred_ents}
        else:
            gold_set = {_normalize(e["text"]) for e in gold_ents}
            pred_set = {_normalize(e["text"]) for e in pred_ents}

        tp_pairs = gold_set & pred_set
        fp_pairs = pred_set - gold_set
        fn_pairs = gold_set - pred_set

        # Per-type accounting (requires exact_type=True)
        if exact_type:
            for text, etype in tp_pairs:
                per_type[etype].true_positives += 1
                overall.true_positives += 1
            for text, etype in fp_pairs:
                per_type[etype].false_positives += 1
                overall.false_positives += 1
            for text, etype in fn_pairs:
                per_type[etype].false_negatives += 1
                overall.false_negatives += 1
        else:
            overall.true_positives  += len(tp_pairs)
            overall.false_positives += len(fp_pairs)
            overall.false_negatives += len(fn_pairs)

    return EvaluationReport(per_type=dict(per_type), overall=overall)
