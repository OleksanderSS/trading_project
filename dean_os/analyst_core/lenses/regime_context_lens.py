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

from typing import Any

from dean_os.analyst_core.lens_contract import AnalysisPacket, AnalystLens, ModuleDelta
from dean_os.analyst_core.schemas import (
    REGIME_DIMENSIONS,
    Confidence,
    RegimeContextVector,
    RegimeDimensionState,
    Trend,
)

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
    "war_escalation": "geopolitical_state",
    "de_escalation": "geopolitical_state",
    "sanctions_change": "geopolitical_state",
    "central_bank_decision": "liquidity_credit_context",
    "inflation_release": "inflation_rates_context",
    "recession_risk": "economic_phase",
    "expansion_signal": "economic_phase",
    "ai_capex_announcement": "ai_tech_cycle",
    "memory_supply_constraint": "ai_tech_cycle",
    "power_grid_constraint": "ai_tech_cycle",
    "commodity_supply_shock": "commodity_stress",
    "oil_shock": "commodity_stress",
    "risk_on_rotation": "market_state",
    "risk_off_rotation": "market_state",
    "safe_haven_bid": "safe_haven_behavior",
    "strategic_industrial_asset_mna": "geopolitical_state",
}

# Default conservative state labels per dimension when no event speaks to it.
DEFAULT_DIMENSION_STATE: dict[str, str] = {
    "geopolitical_state": "stable",
    "economic_phase": "unknown",
    "inflation_rates_context": "unknown",
    "liquidity_credit_context": "neutral",
    "market_state": "unknown",
    "commodity_stress": "low",
    "ai_tech_cycle": "early_adoption",
    "safe_haven_behavior": "none",
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

        # Walk events: each event nudges its mapped dimension.
        touched_dimensions: set[str] = set()
        for event in packet.event_records:
            event_class = str(event.get("event_class") or "").strip().lower()
            dimension = EVENT_CLASS_TO_DIMENSION.get(event_class)
            if dimension is None:
                continue
            touched_dimensions.add(dimension)

            state = str(event.get("regime_state") or DEFAULT_DIMENSION_STATE.get(dimension, "unknown"))
            # Intensity: derive from event strength if provided, else moderate.
            intensity = self._clamp(float(event.get("intensity", 0.5)))
            trend = self._parse_trend(event.get("trend"))
            evidence_ids = self._collect_evidence_ids(event, evidence_by_id)
            dimension_evidence[dimension].extend(evidence_ids)

            dimensions[dimension] = RegimeDimensionState(
                state=state,
                intensity=intensity,
                trend=trend,
                confidence=Confidence.MEDIUM if evidence_ids else Confidence.LOW,
                evidence_ids=dimension_evidence[dimension],
                notes=str(event.get("summary") or ""),
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
            as_of=packet.as_of_date,
            dimensions=dimensions,
            confidence=overall_confidence,
            evidence_gaps=self._evidence_gaps_for_untouched(touched_dimensions),
        )

        return ModuleDelta(
            module_name=self.lens_name,
            module_version=self.lens_version,
            regime_context=regime,
            fields_added=["regime_context"],
            evidence_ids=sorted({eid for evs in dimension_evidence.values() for eid in evs}),
            confidence=0.5 if overall_confidence == Confidence.MEDIUM else 0.3,
            reason_for_change=(
                f"Graded regime vector from {len(packet.event_records)} events "
                f"(touched {len(touched_dimensions)}/8 dimensions)."
            ),
        )

    # ── helpers ──────────────────────────────────────────────────────────────

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


__all__ = ["RegimeContextLens", "EVENT_CLASS_TO_DIMENSION", "DEFAULT_DIMENSION_STATE"]
