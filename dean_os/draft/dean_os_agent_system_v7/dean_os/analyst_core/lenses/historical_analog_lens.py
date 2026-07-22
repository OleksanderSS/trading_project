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

from dean_os.analyst_core.lens_contract import AnalysisPacket, AnalystLens, ModuleDelta

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
        "analog_id": "ai_bubble_dotcom",
        "pattern_name": "technology_valuation_bubble",
        "trigger_keywords": ("bubble", "valuation", "overvalued", "speculative"),
        "sector_keywords": ("technology", "ai", "semiconductor", "cloud"),
        "event_similarity": 0.5,
        "regime_similarity": 0.3,
        "key_similarities": [
            "technology-driven investment cycle",
            "high valuations relative to earnings",
            "broad market enthusiasm",
        ],
        "key_differences": [
            "dotcom had no real revenue; AI companies have cash flows",
            "AI infrastructure spending is more concentrated",
            "different interest rate environment",
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
            "AI leaders have real earnings and cash flows. Timeline and "
            "magnitude may differ significantly."
        ),
    },
    {
        "analog_id": "tariff_2018",
        "pattern_name": "trade_war_escalation",
        "trigger_keywords": ("tariff", "trade war", "customs", "duty"),
        "sector_keywords": ("semiconductor", "industrial", "consumer", "technology"),
        "event_similarity": 0.65,
        "regime_similarity": 0.45,
        "key_similarities": [
            "bilateral tariff escalation",
            "supply chain disruption fears",
            "market volatility spike",
        ],
        "key_differences": [
            "2018 was US-China bilateral, current may involve more parties",
            "semiconductor export controls are newer dimension",
        ],
        "outcomes_by_horizon": {
            1: {"direction": "uncertainty_selloff", "return_pct": -2.8},
            5: {"direction": "partial_recovery", "return_pct": 1.3},
            20: {"direction": "trade_deal_hope", "return_pct": 4.7},
            60: {"direction": "deal_fatigue", "return_pct": -3.2},
            120: {"direction": "structural_deglobalization", "return_pct": -8.5},
        },
        "winning_scenario": "escalation_then_negotiated_deal",
        "false_analogy_risk": "medium",
        "why_this_case_may_mislead": (
            "2018 ended in a deal. Current trade tensions involve "
            "technology restrictions that may be structural, not negotiable."
        ),
    },
    {
        "analog_id": "fed_hike_2022",
        "pattern_name": "aggressive_rate_hiking_cycle",
        "trigger_keywords": ("rate hike", "hawkish", "tightening", "fomc"),
        "sector_keywords": ("financials", "technology", "consumer", "real estate"),
        "event_similarity": 0.55,
        "regime_similarity": 0.4,
        "key_similarities": [
            "rapid rate increases from near-zero",
            "inflation-driven policy response",
            "growth stock compression",
        ],
        "key_differences": [
            "2022 was from zero-lower-bound; current starting point differs",
            "inflation drivers may be different",
        ],
        "outcomes_by_horizon": {
            1: {"direction": "rate_shock_selloff", "return_pct": -3.5},
            5: {"direction": "continued_pressure", "return_pct": -5.2},
            20: {"direction": "pivot_hope_rally", "return_pct": 8.1},
            60: {"direction": "soft_landing_narrative", "return_pct": 12.3},
            120: {"direction": "normalized_rates", "return_pct": 18.7},
        },
        "winning_scenario": "aggressive_hikes_then_pivot",
        "false_analogy_risk": "medium",
        "why_this_case_may_mislead": (
            "2022-23 hiking cycle ended with pivot expectations. "
            "Future cycles may have different duration and terminal rate."
        ),
    },
]


class HistoricalAnalogLens(AnalystLens):
    """Retrieves historical analogs for classified events.

    Matches classified events against the analog library and produces
    ``watch_signals`` with analog candidates including outcomes and
    false-analogy warnings.
    """

    lens_name = "historical_analog"
    lens_version = "0.1.0"
    event_classes_supported = ("*",)
    can_modify_existing = False

    def analyze(
        self, packet: AnalysisPacket, config: dict[str, Any] | None = None
    ) -> ModuleDelta:
        config = config or {}
        max_analogs = config.get("max_analogs_per_event", 3)
        analog_signals: list[dict[str, Any]] = []

        all_events = list(packet.entity_links) + list(packet.event_records)
        for event in all_events:
            if not isinstance(event, dict):
                continue
            matches = self._find_analogs(event, max_analogs)
            analog_signals.extend(matches)

        review_notes: list[str] = []
        if not analog_signals:
            review_notes.append(
                "historical_analog: no analog matches found for current events"
            )

        return ModuleDelta(
            module_name=self.lens_name,
            module_version=self.lens_version,
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

    def _find_analogs(
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
            sum(s["event_similarity"] for s in signals) / len(signals)
        )
        high_risk = sum(
            1 for s in signals if s["false_analogy_risk"] == "high"
        )
        penalty = high_risk * 0.1
        return max(0.2, min(0.8, 0.3 + avg_similarity * 0.4 - penalty))


__all__ = ["HistoricalAnalogLens", "ANALOG_LIBRARY"]
