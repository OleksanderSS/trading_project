"""RegimeContextLens — grades the daily regime vector (note 03 §6-7, note 07 §1).

This is the FIRST concrete lens and exists to prove the plugin pattern end to
end: it reads an AnalysisPacket (events + evidence), produces a
``RegimeContextVector``, and returns it as a ``ModuleDelta``. It is fully
deterministic — no LLM, no network — and review-only.

The grading is intentionally conservative and rule-based. It maps event
classes + evidence stances onto the eight regime dimensions defined in
``REGIME_DIMENSIONS``. Real-world regime grading will later be enriched by a
domain KnowledgePack and (optionally) an LLM interpretation lens that runs
AFTER this deterministic baseline. The deterministic baseline is never removed:
it is the auditable fallback the system falls back to when richer signals are
absent or untrusted.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from dean_os.analyst_core.lens_contract import AnalysisPacket, AnalystLens, ModuleDelta
from dean_os.analyst_core.schemas import (
    REGIME_DIMENSIONS,
    Confidence,
    RegimeContextVector,
    RegimeDimensionState,
    Trend,
)
from dean_os.utils import sha256_json

# ──────────────────────────────────────────────────────────────────────────────
# Event class → regime dimension mapping (note 04 §7 routing table).
#
# When a lens sees an event of a given class, it nudges the corresponding
# regime dimension. This is deliberately a coarse, conservative mapping: it
# establishes that the event is RELEVANT to a dimension without overclaiming
# the resulting state. The actual `state` string is taken from the event
# record if present, else defaulted.
# ──────────────────────────────────────────────────────────────────────────────

EVENT_CLASS_TO_DIMENSION: dict[str, str] = {
    # Geopolitical
    "war_escalation": "geopolitical_state",
    "de_escalation": "geopolitical_state",
    "sanctions_change": "geopolitical_state",
    "tariff": "geopolitical_state",
    "regulation": "geopolitical_state",
    "strategic_industrial_asset_mna": "geopolitical_state",
    "political_transition": "geopolitical_state",
    "trade_route_disruption": "geopolitical_state",
    # Monetary / Liquidity
    "central_bank_decision": "liquidity_credit_context",
    "liquidity_observation": "liquidity_credit_context",
    "debt_crisis": "liquidity_credit_context",
    # Inflation
    "inflation_release": "inflation_rates_context",
    "inflation_observation": "inflation_rates_context",
    # Economic phase
    "recession_risk": "economic_phase",
    "expansion_signal": "economic_phase",
    "pandemic_health_shock": "economic_phase",
    # AI / Tech cycle
    "ai_capex_announcement": "ai_tech_cycle",
    "demand_driver": "ai_tech_cycle",
    "supply_disruption": "ai_tech_cycle",
    "capex_signal": "ai_tech_cycle",
    "memory_supply_constraint": "ai_tech_cycle",
    "power_grid_constraint": "ai_tech_cycle",
    # Commodity stress
    "commodity_supply_shock": "commodity_stress",
    "oil_shock": "commodity_stress",
    "climate_disaster": "commodity_stress",  # physical disruption hits energy/commodity prices
    # Market state
    "risk_on_rotation": "market_state",
    "risk_off_rotation": "market_state",
    "sector_rotation": "market_state",
    # Safe haven
    "safe_haven_bid": "safe_haven_behavior",
}

# Macro context_key -> regime dimension (dean_os/config/macro_series_registry.yaml
# is the source of truth for which FRED series exist and their context_key).
# Without this, macro evidence from SavedMacroEvidenceProducer +
# MarketContextEvidenceAdapter reached this lens with event_class values like
# "rates_policy"/"inflation"/"market_confirmation"
# (MarketContextEvidenceAdapter.MACRO_SERIES_EVIDENCE_MAP -- a coarser,
# domain-relevance-filtering concern) that don't appear anywhere in
# EVENT_CLASS_TO_DIMENSION above, so every macro observation was silently
# dropped (`if dimension is None: continue`). This maps each series directly
# to the dimension it actually speaks to. Series with no honest
# single-dimension meaning (FX pairs) are left unmapped rather than guessed.
#
# NOTE: keyed by context_key, not the raw FRED series id (e.g. "fed_funds_rate",
# not "FEDFUNDS"). SavedMacroEvidenceProducer's market_context_fragment.macro
# dict -- and therefore audit_structured_context's observation["name"], which
# is what actually reaches this lens via event["provenance"]["name"] -- is
# keyed by context_key. A first version of this map used the raw series id
# and silently matched nothing against live data; caught by an end-to-end
# smoke test against a real SavedMacroEvidenceProducer artifact, not by the
# hand-built unit tests, which had (wrongly) assumed the series-id shape.
MACRO_SERIES_TO_DIMENSION: dict[str, str] = {
    "high_yield_oas": "liquidity_credit_context",  # BAMLH0A0HYM2: credit stress
    "continued_claims": "economic_phase",  # CCSA
    "cpi": "inflation_rates_context",  # CPIAUCSL
    "wti_crude_oil": "commodity_stress",  # DCOILWTICO
    "durable_goods_orders": "economic_phase",  # DGORDER
    "treasury_10y_daily": "liquidity_credit_context",  # DGS10
    "real_disposable_personal_income": "economic_phase",  # DSPIC96
    "fed_funds_rate": "liquidity_credit_context",  # FEDFUNDS
    "treasury_10y_monthly": "liquidity_credit_context",  # GS10
    "treasury_2y_monthly": "liquidity_credit_context",  # GS2
    "housing_starts": "economic_phase",  # HOUST
    "industrial_production": "economic_phase",  # INDPRO
    "manufacturing_employment": "economic_phase",  # MANEMP
    "nonfarm_payrolls": "economic_phase",  # PAYEMS
    "pce_price_index": "inflation_rates_context",  # PCEPI
    "core_pce_price_index": "inflation_rates_context",  # PCEPILFE
    "building_permits": "economic_phase",  # PERMIT
    "producer_price_index": "inflation_rates_context",  # PPIACO
    "retail_sales": "economic_phase",  # RSAFS
    "yield_curve_10y_2y": "liquidity_credit_context",  # T10Y2Y
    "total_vehicle_sales": "economic_phase",  # TOTALSA
    "consumer_sentiment": "economic_phase",  # UMCSENT
    "unemployment_rate": "economic_phase",  # UNRATE
    "vix": "safe_haven_behavior",  # VIXCLS
    "federal_reserve_total_assets": "liquidity_credit_context",  # WALCL
    # usd_cny, usd_eur (DEXCHUS/DEXUSEU): FX pairs have no single honest
    # regime meaning without more context (trade policy vs growth vs risk
    # sentiment) -- left unmapped rather than guessed.
}
EVENT_CLASS_TO_DIMENSION.update(MACRO_SERIES_TO_DIMENSION)

# Single-snapshot numeric thresholds for the two series where ONE observation
# (no history needed) is still enough to grade direction, using thresholds
# that are standard market convention, not invented cutoffs: T10Y2Y < 0 is
# the textbook yield-curve-inversion recession signal; VIX >= 25 / <= 15 are
# the commonly cited elevated-fear / complacency bands. Every other mapped
# macro series still routes to its dimension (via the context_key keys added
# above) but keeps trend=UNKNOWN -- a single snapshot value can't honestly
# say "rising" or "falling" without a prior observation to compare against.
_YIELD_CURVE_INVERSION_THRESHOLD = 0.0
_VIX_FEAR_THRESHOLD = 25.0
_VIX_COMPLACENCY_THRESHOLD = 15.0

_MACRO_SYNTHETIC_CLASS_TO_DIMENSION: dict[str, str] = {
    "yield_curve_inverted": "liquidity_credit_context",
    "yield_curve_normal": "liquidity_credit_context",
    "vix_elevated_fear": "safe_haven_behavior",
    "vix_complacent": "safe_haven_behavior",
    "vix_neutral": "safe_haven_behavior",
}
EVENT_CLASS_TO_DIMENSION.update(_MACRO_SYNTHETIC_CLASS_TO_DIMENSION)

_MACRO_SYNTHETIC_CLASS_TREND: dict[str, str] = {
    "yield_curve_inverted": "falling",
    "yield_curve_normal": "stable",
    "vix_elevated_fear": "rising",
    "vix_complacent": "falling",
    "vix_neutral": "stable",
}


def _macro_event_class_override(context_key: str | None, value: float | None) -> str | None:
    """A more specific event_class for series with a defensible single-snapshot
    threshold; the bare context_key for every other mapped macro series --
    still routes correctly, just without a directional read. None if the
    series isn't mapped at all (e.g. an FX pair)."""
    if not context_key:
        return None
    if context_key == "yield_curve_10y_2y" and value is not None:
        return "yield_curve_inverted" if value < _YIELD_CURVE_INVERSION_THRESHOLD else "yield_curve_normal"
    if context_key == "vix" and value is not None:
        if value >= _VIX_FEAR_THRESHOLD:
            return "vix_elevated_fear"
        if value <= _VIX_COMPLACENCY_THRESHOLD:
            return "vix_complacent"
        return "vix_neutral"
    if context_key in MACRO_SERIES_TO_DIMENSION:
        return context_key
    return None


# Default conservative state labels per dimension when no event speaks to it.
DEFAULT_DIMENSION_STATE: dict[str, str] = {
    "geopolitical_state": "unknown",
    "economic_phase": "unknown",
    "inflation_rates_context": "unknown",
    "liquidity_credit_context": "unknown",
    "market_state": "unknown",
    "commodity_stress": "unknown",
    "ai_tech_cycle": "unknown",
    "safe_haven_behavior": "unknown",
}


class RegimeContextLens(AnalystLens):
    """Deterministic regime-context grader.

    Produces a ``RegimeContextVector`` from the packet's events + evidence.
    Never raises on missing inputs — it degrades to a low-confidence, all-
    "unknown" vector so downstream lenses can still reason about shape.
    """

    lens_name = "regime_context"
    lens_version = "0.1.0"

    # This lens establishes the regime baseline; it runs for every event class.
    event_classes_supported = ("*",)
    can_modify_existing = False

    def analyze(self, packet: AnalysisPacket, config: dict[str, Any] | None = None) -> ModuleDelta:
        config = config or {}
        evidence_by_id = {e.get("evidence_id"): e for e in self._evidence_lookup(packet)}

        # Start every dimension at its conservative default.
        dimensions: dict[str, RegimeDimensionState] = {}
        dimension_evidence: dict[str, list[str]] = {dim: [] for dim in DEFAULT_DIMENSION_STATE}

        events = [
            self._apply_macro_grading(event)
            for event in (packet.classified_events or packet.event_records)
        ]

        # Aggregate first. A dimension may receive many evidence items, so its
        # state must not depend on whichever record happened to be last.
        touched_dimensions: set[str] = set()
        events_by_dimension: dict[str, list[dict[str, Any]]] = {
            dim: [] for dim in DEFAULT_DIMENSION_STATE
        }
        for event in events:
            event_class = str(event.get("event_class") or "").strip().lower()
            dimension = EVENT_CLASS_TO_DIMENSION.get(event_class)
            if dimension is None:
                continue
            touched_dimensions.add(dimension)
            events_by_dimension[dimension].append(event)
            evidence_ids = self._collect_evidence_ids(event, evidence_by_id)
            dimension_evidence[dimension].extend(evidence_ids)

        for dimension in sorted(touched_dimensions):
            dimension_events = events_by_dimension[dimension]
            class_counts = Counter(
                str(event.get("event_class") or "other")
                for event in dimension_events
            )
            dominant_class = sorted(
                class_counts,
                key=lambda name: (-class_counts[name], name),
            )[0]
            state = (
                f"{dominant_class}_signal"
                if len(class_counts) == 1
                else "mixed_signals"
            )
            intensity_values = [
                self._clamp(
                    float(
                        event.get("intensity")
                        or event.get("materiality_score")
                        or event.get("strength")
                        or 0.0
                    )
                )
                for event in dimension_events
            ]
            intensity = (
                sum(intensity_values) / len(intensity_values)
                if intensity_values
                else 0.0
            )
            trends = {
                self._parse_trend(event.get("trend"))
                for event in dimension_events
            }
            trend = trends.pop() if len(trends) == 1 else Trend.UNKNOWN
            evidence_ids = sorted(set(dimension_evidence[dimension]))
            signal_summary = ", ".join(
                f"{name}:{class_counts[name]}"
                for name in sorted(class_counts)
            )
            dimensions[dimension] = RegimeDimensionState(
                state=state,
                intensity=intensity,
                trend=trend,
                confidence=(
                    Confidence.MEDIUM if evidence_ids else Confidence.LOW
                ),
                evidence_ids=evidence_ids,
                notes=f"Evidence-backed signal counts: {signal_summary}.",
            )

        # Fill untouched dimensions with conservative defaults.
        for dim, default_state in DEFAULT_DIMENSION_STATE.items():
            if dim not in dimensions:
                dimensions[dim] = RegimeDimensionState(
                    state=default_state,
                    intensity=0.0,
                    trend=Trend.UNKNOWN,
                    confidence=Confidence.LOW,
                    evidence_ids=[],
                )

        overall_confidence = Confidence.MEDIUM if touched_dimensions else Confidence.LOW

        regime = RegimeContextVector(
            regime_context_id="regime_" + sha256_json(
                {
                    "as_of": packet.as_of_date,
                    "dimensions": {
                        key: value.model_dump(mode="json")
                        for key, value in dimensions.items()
                    },
                }
            )[:24],
            as_of=packet.as_of_date,
            dimensions=dimensions,
            confidence=overall_confidence,
            evidence_gaps=self._evidence_gaps_for_untouched(touched_dimensions),
        )

        return ModuleDelta(
            module_name=self.lens_name,
            module_version=self.lens_version,
            as_of=packet.as_of_date,
            regime_context=regime,
            fields_added=["regime_context"],
            evidence_ids=sorted({eid for evs in dimension_evidence.values() for eid in evs}),
            confidence=0.5 if overall_confidence == Confidence.MEDIUM else 0.3,
            reason_for_change=(
                f"Graded regime vector from {len(events)} classified events "
                f"(touched {len(touched_dimensions)}/8 dimensions)."
            ),
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _apply_macro_grading(event: dict[str, Any]) -> dict[str, Any]:
        """For macro-sourced events (FRED observations via
        SavedMacroEvidenceProducer + MarketContextEvidenceAdapter), replace the
        generic evidence_type event_class (e.g. "rates_policy") with the
        specific registry context_key -- or a numeric-threshold-graded label
        for yield_curve_10y_2y/vix -- so routing and state-naming reflect what
        was actually observed instead of the coarse domain-relevance bucket.
        Non-macro events (news) and unmapped series pass through unchanged.
        Never mutates the input event.
        """
        if event.get("source_type") != "macro":
            return event
        provenance = event.get("provenance") or {}
        # audit_structured_context keys macro observations by context_key
        # (SavedMacroEvidenceProducer's market_context_fragment.macro dict),
        # not the raw FRED series id -- observation["name"] carries that key.
        context_key = provenance.get("name")
        raw_value = provenance.get("value")
        try:
            value = float(raw_value) if raw_value is not None else None
        except (TypeError, ValueError):
            value = None
        override = _macro_event_class_override(context_key, value)
        if override is None:
            return event
        graded = dict(event)
        graded["event_class"] = override
        if not graded.get("trend"):
            graded["trend"] = _MACRO_SYNTHETIC_CLASS_TREND.get(override)
        return graded

    @staticmethod
    def _evidence_lookup(packet: AnalysisPacket) -> list[dict[str, Any]]:
        """Evidence items attached to the packet (free-form dicts for now)."""
        # The packet currently carries evidence via entity_links/event_records;
        # a dedicated evidence field will land in Phase 2. Read defensively.
        raw = getattr(packet, "evidence", None)
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
        return []

    @staticmethod
    def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, value))

    @staticmethod
    def _parse_trend(value: Any) -> Trend:
        if isinstance(value, Trend):
            return value
        text = str(value or "").strip().lower()
        for member in Trend:
            if member.value == text:
                return member
        return Trend.UNKNOWN

    @staticmethod
    def _collect_evidence_ids(event: dict[str, Any], evidence_by_id: dict[Any, dict]) -> list[str]:
        ids: list[str] = []
        raw_ids = event.get("evidence_ids") or event.get("evidence_id")
        if isinstance(raw_ids, str):
            ids.append(raw_ids)
        elif isinstance(raw_ids, (list, tuple)):
            ids.extend(str(item) for item in raw_ids)
        return sorted({eid for eid in ids if eid})

    @staticmethod
    def _evidence_gaps_for_untouched(touched_dimensions: set[str]) -> list[str]:
        """Human-readable gap labels for dimensions no event spoke to.

        These are lightweight string hints (note 03 §7 ``evidence_gaps`` field).
        Phase 2's evidence_gap lens will promote them to structured
        ``EvidenceGap`` objects ranked by scenario-probability importance.
        """
        untouched = set(REGIME_DIMENSIONS) - touched_dimensions
        if not untouched:
            return []
        return sorted(f"no_evidence_for_{dim}" for dim in untouched)


__all__ = [
    "RegimeContextLens",
    "EVENT_CLASS_TO_DIMENSION",
    "DEFAULT_DIMENSION_STATE",
    "MACRO_SERIES_TO_DIMENSION",
]
