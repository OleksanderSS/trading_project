"""HistoricalAnalogLens — retrieves historical analogs with outcome tracking.

This lens finds structurally similar past events and records what actually
happened at fixed outcome horizons (1d, 5d, 20d, 60d, 120d). It produces
``watch_signals`` entries with analog candidates.

Every analog MUST include ``why_this_case_may_mislead`` to prevent
narrative overfitting (from design notes §3 §11).

Deterministic pattern matching. No LLM, no network.
"""
from __future__ import annotations

from typing import Any
import json
from pathlib import Path

from dean_os.analyst_core.lens_contract import AnalysisPacket, AnalystLens, ModuleDelta
from dean_os.analyst_core.utils.knn_similarity_finder import KnnSimilarityFinder

# ──────────────────────────────────────────────────────────────────────────────
# Historical analog library (seed from design notes, §3 §11)
# Each entry represents a structural pattern that has occurred before.
# ──────────────────────────────────────────────────────────────────────────────

ANALOG_LIBRARY: list[dict[str, Any]] = [
    {
        "analog_id": "chipshortage_2021",
        "pattern_name": "semiconductor_supply_shortage",
        "trigger_keywords": ("shortage", "supply disruption", "bottleneck", "allocation"),
        "sector_keywords": ("semiconductor", "chip", "foundry", "auto", "industrial"),
        "event_similarity": 0.7,
        "regime_similarity": 0.5,
        "key_similarities": [
            "demand surge exceeding supply capacity",
            "long lead times for critical components",
            "downstream production impacts",
        ],
        "key_differences": [
            "2021 was post-covid demand spike, current may be AI-driven",
            "auto sector was primary victim in 2021",
        ],
        "outcomes_by_horizon": {
            1: {"direction": "positive_for_semiconductors", "return_pct": 2.5},
            5: {"direction": "continued_strength", "return_pct": 4.1},
            20: {"direction": "supply_appeasement_concerns", "return_pct": -1.2},
            60: {"direction": "normalization_pricing", "return_pct": -5.3},
            120: {"direction": "cyclical_peak_recognition", "return_pct": -12.1},
        },
        "winning_scenario": "supply_shortage_initially_bullish_then_mean_reversion",
        "false_analogy_risk": "medium",
        "why_this_case_may_mislead": (
            "2021 shortage was demand-driven; current AI capex may have "
            "different duration and customer concentration profile."
        ),
    },
    {
        "analog_id": "oil_embargo_2022",
        "pattern_name": "energy_shock_geopolitical",
        "trigger_keywords": ("oil spike", "energy shock", "embargo", "supply cut"),
        "sector_keywords": ("energy", "oil", "gas", "industrial", "transport"),
        "event_similarity": 0.6,
        "regime_similarity": 0.4,
        "key_similarities": [
            "geopolitical trigger for energy supply disruption",
            "inflationary transmission through economy",
            "safe haven demand spike",
        ],
        "key_differences": [
            "2022 was Russia-Ukraine, different geopolitical context",
            "current OPEC dynamics may differ",
        ],
        "outcomes_by_horizon": {
            1: {"direction": "energy_up_equities_down", "return_pct": -1.8},
            5: {"direction": "continued_volatility", "return_pct": -3.2},
            20: {"direction": "partial_recovery", "return_pct": 1.5},
            60: {"direction": "demand_destruction", "return_pct": -8.7},
            120: {"direction": "structural_repricing", "return_pct": -15.3},
        },
        "winning_scenario": "energy_shock_transitory_then_structural",
        "false_analogy_risk": "medium",
        "why_this_case_may_mislead": (
            "Each energy shock has unique supply/demand structure. "
            "2022 was sanctions-driven; current may be production-cut-driven."
        ),
    },
    {
        "analog_id": "dotcom_bubble_1999",
        "pattern_name": "tech_infrastructure_overbuild",
        "trigger_keywords": ("bubble", "overbuild", "capex", "valuation"),
        "sector_keywords": ("technology", "semiconductor", "telecom"),
        "event_similarity": 0.6,
        "regime_similarity": 0.5,
        "key_similarities": [
            "massive infrastructure build-out ahead of end-user demand",
            "retail speculation in related names",
            "narrative-driven valuations",
        ],
        "key_differences": [
            "AI companies have actual revenue today",
            "interest rate environment differs",
        ],
        "outcomes_by_horizon": {
            1: {"direction": "continued_momentum", "return_pct": 5.2},
            5: {"direction": "profit_taking", "return_pct": -2.1},
            20: {"direction": "correction_began", "return_pct": -12.5},
            60: {"direction": "capitulation", "return_pct": -35.0},
            120: {"direction": "structural_reset", "return_pct": -55.0},
        },
        "winning_scenario": "bubble_pop_then_structural_reset",
        "false_analogy_risk": "high",
        "why_this_case_may_mislead": (
            "Dotcom bubble was unprofitable companies with no revenue. "
            "AI leaders have real earnings and cash flows."
        ),
    },
]


class HistoricalAnalogLens(AnalystLens):
    """Retrieves historical analogs for classified events.

    Matches classified events against historical world states using KNN cluster search.
    If no historical data is available (untrained system), falls back to deterministic seed analog matching.
    """

    lens_name = "historical_analog"
    lens_version = "0.2.0" # upgraded to support KNN
    event_classes_supported = ("*",)
    can_modify_existing = False

    def analyze(
        self, packet: AnalysisPacket, config: dict[str, Any] | None = None
    ) -> ModuleDelta:
        config = config or {}
        max_analogs = config.get("max_analogs_per_event", 3)
        pipeline_context = config.get("pipeline_context", {})
        
        # Load historical contexts
        historical_records = self._load_historical_contexts()
        
        knn_finder = None
        knn_ready = False
        
        # We need both historical records AND a current pipeline context to use KNN
        if historical_records and pipeline_context and pipeline_context.get("indicator_state_grid"):
            knn_finder = KnnSimilarityFinder(n_neighbors=max_analogs)
            # Use the indicator state grid features as historical data
            flat_hist = []
            for rec in historical_records:
                if rec.get("indicator_state_grid"):
                    grid = rec["indicator_state_grid"]
                    flat_rec = {"_as_of": rec.get("as_of", "")}
                    for group_name, variables in grid.get("groups", {}).items():
                        for var_name, var_data in variables.items():
                            val = var_data.get("value")
                            if val is not None:
                                flat_rec[f"{group_name}.{var_name}"] = float(val)
                    flat_hist.append(flat_rec)
            
            if flat_hist:
                knn_finder.fit(flat_hist)
                knn_ready = True
        
        analog_signals: list[dict[str, Any]] = []
        review_notes: list[str] = []

        all_events = list(packet.entity_links) + list(packet.event_records)
        
        for event in all_events:
            if not isinstance(event, dict):
                continue
            
            event_id = event.get("event_id", event.get("id", ""))
            
            if knn_ready and pipeline_context.get("indicator_state_grid"):
                # Use KNN search
                current_features = {}
                grid = pipeline_context["indicator_state_grid"]
                for group_name, variables in grid.get("groups", {}).items():
                    for var_name, var_data in variables.items():
                        val = var_data.get("value")
                        if val is not None:
                            current_features[f"{group_name}.{var_name}"] = float(val)
                
                knn_matches = knn_finder.find_analogies(current_features)
                if knn_matches:
                    for i, match in enumerate(knn_matches):
                        analog_signals.append({
                            "analog_id": f"knn_analog_{match.get('_as_of', 'unknown')}",
                            "pattern_name": f"knn_cluster_{i}",
                            "source_event_id": event_id,
                            "event_similarity": match.get("_knn_similarity", 0.0),
                            "regime_similarity": match.get("_knn_similarity", 0.0),
                            "key_similarities": ["Similar quantitative world-state indicators"],
                            "key_differences": ["Time context"],
                            "outcomes_by_horizon": {
                                "1": {"direction": "unknown_knn_horizon_1", "return_pct": 0.0},
                                "5": {"direction": "unknown_knn_horizon_5", "return_pct": 0.0},
                                "20": {"direction": "unknown_knn_horizon_20", "return_pct": 0.0},
                            },
                            "winning_scenario": "calibrated_base_rate",
                            "false_analogy_risk": "low",
                            "why_this_case_may_mislead": "KNN distance only considers numeric indicators, ignoring qualitative events."
                        })
                    review_notes.append(f"historical_analog: used KNN cluster search for {event_id} based on {len(knn_matches)} neighbors")
                    continue
            
            # Fallback to deterministic matching
            matches = self._find_analogs_deterministic(event, max_analogs)
            analog_signals.extend(matches)
            review_notes.append(f"historical_analog: used deterministic seed matching for {event_id} (KNN not ready)")

        if not analog_signals:
            review_notes.append(
                "historical_analog: no analog matches found for current events"
            )

        return ModuleDelta(
            module_name=self.lens_name,
            module_version=self.lens_version,
            as_of=packet.as_of_date,
            watch_signals_added=[
                {
                    "signal_type": "historical_analog",
                    "analog_id": s["analog_id"],
                    "pattern_name": s["pattern_name"],
                    "event_id": s["source_event_id"],
                    "event_similarity": s["event_similarity"],
                    "false_analogy_risk": s["false_analogy_risk"],
                    "outcome_hint": s.get("winning_scenario", ""),
                }
                for s in analog_signals
            ],
            fields_added=["historical_analogs"],
            confidence=self._overall_confidence(analog_signals),
            reason_for_change=(
                f"Found {len(analog_signals)} analog matches across "
                f"{len(all_events)} events."
            ),
            review_notes_added=review_notes,
        )

    def _load_historical_contexts(self) -> list[dict[str, Any]]:
        """Mock loader for historical contexts. In a fully trained system, this would read from the world state DB."""
        # For a "minimally ready system that is not yet trained", we return an empty list
        # so it safely falls back, but the mechanism is fully in place.
        records = []
        base_dir = Path("reports/dean_os")
        if base_dir.exists():
            for p in base_dir.glob("world_model_pipeline_context_*/latest.json"):
                try:
                    records.append(json.loads(p.read_text(encoding="utf-8")))
                except Exception:
                    pass
        return records

    def _find_analogs_deterministic(
        self, event: dict[str, Any], max_matches: int
    ) -> list[dict[str, Any]]:
        text = self._event_text(event).lower()
        event_class = str(event.get("event_class", "")).strip()

        scored: list[tuple[float, dict[str, Any]]] = []

        for analog in ANALOG_LIBRARY:
            score = self._match_score(text, event_class, analog)
            if score > 0.2:
                scored.append((score, analog))

        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[dict[str, Any]] = []
        for score, analog in scored[:max_matches]:
            results.append({
                "analog_id": analog["analog_id"],
                "pattern_name": analog["pattern_name"],
                "source_event_id": event.get("event_id", event.get("id", "")),
                "event_similarity": round(
                    min(1.0, score * analog["event_similarity"]), 3
                ),
                "regime_similarity": analog["regime_similarity"],
                "key_similarities": list(analog["key_similarities"]),
                "key_differences": list(analog["key_differences"]),
                "outcomes_by_horizon": {
                    str(k): v
                    for k, v in analog["outcomes_by_horizon"].items()
                },
                "winning_scenario": analog["winning_scenario"],
                "false_analogy_risk": analog["false_analogy_risk"],
                "why_this_case_may_mislead": analog["why_this_case_may_mislead"],
            })

        return results

    def _match_score(
        self, text: str, event_class: str, analog: dict[str, Any]
    ) -> float:
        score = 0.0

        # Trigger keyword match
        trigger_hits = sum(
            1 for kw in analog["trigger_keywords"] if kw in text
        )
        score += trigger_hits * 0.15

        # Sector keyword match
        sector_hits = sum(
            1 for kw in analog["sector_keywords"] if kw in text
        )
        score += sector_hits * 0.1

        # Event class alignment (rough)
        if event_class and any(
            kw in event_class for kw in analog["trigger_keywords"]
        ):
            score += 0.2

        return min(1.0, score)

    def _event_text(self, event: dict[str, Any]) -> str:
        parts = []
        for key in ("text", "title", "summary", "text_preview", "description"):
            val = event.get(key)
            if val and isinstance(val, str):
                parts.append(val)
        return " ".join(parts) if parts else str(event)

    def _overall_confidence(self, signals: list[dict[str, Any]]) -> float:
        if not signals:
            return 0.2
        avg_similarity = (
            sum(s.get("event_similarity", 0) for s in signals) / len(signals)
        )
        high_risk = sum(
            1 for s in signals if s.get("false_analogy_risk") == "high"
        )
        penalty = high_risk * 0.1
        return max(0.2, min(0.8, 0.3 + avg_similarity * 0.4 - penalty))


__all__ = ["HistoricalAnalogLens", "ANALOG_LIBRARY"]
