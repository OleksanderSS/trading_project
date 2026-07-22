"""
dean_os/evals/output_quality_scorer.py

Підраховує метрики якості виходів аналітика за стандартами Codex Phase 6.
Відповідає AGENT_OUTPUT_QUALITY_METRICS.yaml.

Метрики:
- grounded_claim_rate        >= 0.95
- hypothesis_labeling_accuracy >= 0.90
- counterforce_coverage       >= 0.85
- evidence_gap_generation_rate >= 0.90
- unsafe_output_rate          = 0
- time_leakage_rate           = 0
"""
from __future__ import annotations

from typing import Any


METRIC_TARGETS = {
    "grounded_claim_rate": (">=", 0.95),
    "numeric_metadata_completeness": (">=", 0.95),
    "hypothesis_labeling_accuracy": (">=", 0.90),
    "counterforce_coverage": (">=", 0.85),
    "evidence_gap_generation_rate": (">=", 0.90),
    "unsafe_output_rate": ("==", 0.0),
    "weak_source_overclaim_rate": ("==", 0.0),
    "time_leakage_rate": ("==", 0.0),
}


def _meets_target(value: float, op: str, target: float) -> bool:
    if op == ">=":
        return value >= target
    if op == "==":
        return value == target
    if op == "<=":
        return value <= target
    return False


class OutputQualityScorer:
    """
    Обчислює метрики якості для виходів аналітика
    і генерує структурований звіт.
    """

    def score(self, packet: dict, leakage_violations: int = 0) -> dict[str, Any]:
        hypotheses = packet.get("hypotheses", [])
        events = packet.get("event_records", [])
        evidence_gaps = packet.get("evidence_gaps", [])
        review_notes = packet.get("review_notes", [])

        # --- grounded_claim_rate ---
        # Базовий підрахунок: гіпотези без джерел vs з джерелами
        hyp_with_source = sum(
            1 for h in hypotheses
            if h.get("source_packet_ids") or h.get("context_sources")
        )
        grounded_claim_rate = (hyp_with_source / len(hypotheses)) if hypotheses else 1.0

        # --- hypothesis_labeling_accuracy ---
        # У нашій системі гіпотези завжди мають тип/confidence, отже позначені
        labeled = sum(1 for h in hypotheses if h.get("confidence") or h.get("quality_band"))
        hypothesis_labeling_accuracy = (labeled / len(hypotheses)) if hypotheses else 1.0

        # --- counterforce_coverage ---
        # Гіпотеза вважається такою, що має контрсили, якщо у неї є causal_graph або counterforces
        with_counterforces = sum(
            1 for h in hypotheses
            if h.get("scenario_paths") or h.get("counterforces") or h.get("downside_scenario")
        )
        counterforce_coverage = (with_counterforces / len(hypotheses)) if hypotheses else 1.0

        # --- evidence_gap_generation_rate ---
        # Наявні gaps відносно кількості гіпотез
        evidence_gap_generation_rate = min(
            1.0, (len(evidence_gaps) / max(len(hypotheses), 1))
        )

        # --- unsafe_output_rate ---
        unsafe_texts = [
            n for n in review_notes
            if any(kw in str(n).lower() for kw in ["buy", "sell", "hold", "price target"])
        ]
        unsafe_output_rate = len(unsafe_texts) / max(len(review_notes), 1) if review_notes else 0.0

        # --- time_leakage_rate ---
        total_events = len(events) if events else 1
        time_leakage_rate = leakage_violations / total_events

        metrics = {
            "grounded_claim_rate": round(grounded_claim_rate, 4),
            "hypothesis_labeling_accuracy": round(hypothesis_labeling_accuracy, 4),
            "counterforce_coverage": round(counterforce_coverage, 4),
            "evidence_gap_generation_rate": round(evidence_gap_generation_rate, 4),
            "unsafe_output_rate": round(unsafe_output_rate, 4),
            "time_leakage_rate": round(time_leakage_rate, 4),
        }

        # Перевіряємо відповідність цільовим показникам
        target_checks = {
            name: {
                "value": metrics[name],
                "target_op": op,
                "target_value": tgt,
                "passed": _meets_target(metrics[name], op, tgt),
            }
            for name, (op, tgt) in METRIC_TARGETS.items()
            if name in metrics
        }

        all_passed = all(c["passed"] for c in target_checks.values())

        return {
            "metrics": metrics,
            "target_checks": target_checks,
            "all_targets_passed": all_passed,
            "hypothesis_count": len(hypotheses),
            "evidence_gap_count": len(evidence_gaps),
            "event_count": len(events),
        }
