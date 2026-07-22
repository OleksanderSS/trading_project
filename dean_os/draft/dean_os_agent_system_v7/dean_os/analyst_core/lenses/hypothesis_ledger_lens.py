"""HypothesisLedgerLens — manages the lifecycle of analyst hypotheses.

This lens creates, confirms, weakens, or falsifies hypotheses based on
new evidence. It produces ``hypotheses`` entries on the packet.

Hypotheses MUST be falsifiable — every hypothesis must carry
``invalidation_signals`` (from design notes §7 §12).

Deterministic, evidence-based updates. No LLM, no network.
"""
from __future__ import annotations

from typing import Any

from dean_os.analyst_core.lens_contract import AnalysisPacket, AnalystLens, ModuleDelta
from dean_os.analyst_core.schemas import (
    OUTCOME_HORIZONS,
    HypothesisLedgerEntry,
    HypothesisStatus,
)

# ──────────────────────────────────────────────────────────────────────────────
# Hypothesis generation rules (from design notes §6.13)
# Maps event classes to candidate hypothesis templates.
# ──────────────────────────────────────────────────────────────────────────────

HYPOTHESIS_TEMPLATES: dict[str, list[dict[str, Any]]] = {
    "war_escalation": [
        {
            "template": "Geopolitical escalation will disrupt affected economic channels over {horizon}",
            "invalidation_signals": [
                "verified ceasefire materially reduces operational risk",
                "trade and logistics flows normalize",
                "risk premia return to pre-escalation levels",
            ],
            "expected_observations": [
                "higher logistics or insurance costs",
                "supply-chain rerouting and risk-premium expansion",
            ],
        },
    ],
    "de_escalation": [
        {
            "template": "De-escalation will reduce regional risk premia over {horizon}",
            "invalidation_signals": [
                "hostilities resume",
                "agreement lacks implementation or monitoring",
                "sanctions and transport constraints remain unchanged",
            ],
            "expected_observations": [
                "lower freight and insurance stress",
                "improving investment and trade expectations",
            ],
        },
    ],
    "inflation_release": [
        {
            "template": "The inflation surprise will alter rates and valuation expectations over {horizon}",
            "invalidation_signals": [
                "subsequent inflation components reverse the surprise",
                "central-bank guidance is unchanged",
                "market-implied rates reject the initial move",
            ],
            "expected_observations": [
                "repricing of policy-rate expectations",
                "rotation between duration-sensitive and defensive assets",
            ],
        },
    ],
    "recession_risk": [
        {
            "template": "Growth deterioration will weaken cyclical demand over {horizon}",
            "invalidation_signals": [
                "leading indicators recover broadly",
                "credit conditions ease without renewed inflation",
                "earnings revisions turn positive across cyclicals",
            ],
            "expected_observations": [
                "weaker orders and utilization",
                "defensive rotation and wider credit spreads",
            ],
        },
    ],
    "power_grid_constraint": [
        {
            "template": "Power and grid constraints will limit infrastructure deployment over {horizon}",
            "invalidation_signals": [
                "firm power capacity comes online ahead of demand",
                "interconnection queues shorten materially",
                "efficiency gains offset incremental load",
            ],
            "expected_observations": [
                "project delays or regional capacity rationing",
                "higher power procurement and infrastructure costs",
            ],
        },
    ],
    "risk_on_rotation": [
        {
            "template": "Risk-on positioning will persist over {horizon}",
            "invalidation_signals": [
                "market breadth narrows further",
                "credit spreads widen",
                "earnings or liquidity expectations deteriorate",
            ],
            "expected_observations": [
                "broader participation in cyclical assets",
                "stable or falling implied volatility",
            ],
        },
    ],
    "risk_off_rotation": [
        {
            "template": "Risk-off positioning will pressure cyclical assets over {horizon}",
            "invalidation_signals": [
                "credit and volatility stress reverse",
                "policy support materially improves liquidity",
                "earnings expectations stabilize",
            ],
            "expected_observations": [
                "defensive and liquidity preference",
                "wider cross-asset risk premia",
            ],
        },
    ],
    "safe_haven_bid": [
        {
            "template": "Safe-haven demand will remain elevated over {horizon}",
            "invalidation_signals": [
                "geopolitical and financial stress indicators normalize",
                "real yields move sharply against the haven asset",
                "funding markets show sustained improvement",
            ],
            "expected_observations": [
                "persistent demand for defensive stores of value",
                "relative weakness in high-beta exposures",
            ],
        },
    ],
    "strategic_industrial_asset_mna": [
        {
            "template": "Strategic industrial consolidation will alter capacity and bargaining power over {horizon}",
            "invalidation_signals": [
                "transaction is blocked or abandoned",
                "integration synergies fail to materialize",
                "new competing capacity offsets concentration",
            ],
            "expected_observations": [
                "changes in capex and capacity allocation",
                "supplier or customer bargaining-power shifts",
            ],
        },
    ],
    "regulation": [
        {
            "template": "The regulatory change will alter costs or market access over {horizon}",
            "invalidation_signals": [
                "implementation is delayed or narrowed",
                "legal challenge suspends enforcement",
                "firms adapt without material cost or access effects",
            ],
            "expected_observations": [
                "compliance investment or product changes",
                "revised market-access and margin expectations",
            ],
        },
    ],
    "earnings_surprise": [
        {
            "template": "The earnings surprise will drive estimate revisions over {horizon}",
            "invalidation_signals": [
                "surprise is non-recurring or accounting-driven",
                "forward guidance contradicts the reported quarter",
                "industry data does not corroborate the result",
            ],
            "expected_observations": [
                "analyst estimate revisions",
                "peer read-through and valuation repricing",
            ],
        },
    ],
    "sector_rotation": [
        {
            "template": "The sector rotation will persist over {horizon}",
            "invalidation_signals": [
                "relative strength reverses without fundamental confirmation",
                "positioning becomes crowded while revisions weaken",
                "macro regime shifts against the sector",
            ],
            "expected_observations": [
                "sustained relative strength and breadth",
                "supportive earnings or flow confirmation",
            ],
        },
    ],
    "demand_driver": [
        {
            "template": "AI demand growth will accelerate through {horizon}",
            "invalidation_signals": [
                "orders fail to appear in backlog",
                "hyperscaler capex guidance cut",
                "enterprise AI ROI remains unproven",
            ],
            "expected_observations": [
                "increasing GPU order lead times",
                "rising data center revenue",
            ],
        },
    ],
    "supply_disruption": [
        {
            "template": "Supply constraints will persist for {horizon}",
            "invalidation_signals": [
                "capacity expansion completed ahead of schedule",
                "demand destruction exceeds supply shortfall",
                "alternative suppliers emerge",
            ],
            "expected_observations": [
                "extended lead times",
                "pricing power for suppliers",
            ],
        },
    ],
    "capex_signal": [
        {
            "template": "Capex cycle will sustain industry growth through {horizon}",
            "invalidation_signals": [
                "capex guidance revised downward",
                "overcapacity concerns emerge",
                "ROI on new capacity disappoints",
            ],
            "expected_observations": [
                "rising equipment orders",
                "foundry utilization above 80%",
            ],
        },
    ],
    "tariff": [
        {
            "template": "Tariff escalation will reshape supply chains over {horizon}",
            "invalidation_signals": [
                "tariff exemption granted",
                "supply chain diversification faster than expected",
                "demand remains inelastic to tariff impact",
            ],
            "expected_observations": [
                "production relocation announcements",
                "price increases passed to consumers",
            ],
        },
    ],
    "oil_shock": [
        {
            "template": "Energy cost increase will feed into broader inflation over {horizon}",
            "invalidation_signals": [
                "strategic reserves released",
                "OPEC reverses production cut",
                "recession destroys demand before inflation manifests",
            ],
            "expected_observations": [
                "rising CPI energy component",
                "transportation cost increases",
            ],
        },
    ],
    "central_bank_decision": [
        {
            "template": "Monetary policy will materially affect liquidity conditions over {horizon}",
            "invalidation_signals": [
                "inflation data surprises to upside",
                "employment data weakens sharply",
                "financial stability concerns emerge",
            ],
            "expected_observations": [
                "forward guidance confirms path",
                "market pricing aligns with guidance",
            ],
        },
    ],
    "sanctions_change": [
        {
            "template": "Sanctions will constrain market access for {horizon}",
            "invalidation_signals": [
                "sanctions enforcement weakened",
                "alternative supply channels established",
                "sanctions reversed or reduced",
            ],
            "expected_observations": [
                "revenue decline in targeted companies",
                "supply chain restructuring announcements",
            ],
        },
    ],
}


HYPOTHESIS_EVENT_CLASS_ALIASES: dict[str, str] = {
    "expansion_signal": "demand_driver",
    "ai_capex_announcement": "capex_signal",
    "memory_supply_constraint": "supply_disruption",
    "commodity_supply_shock": "supply_disruption",
}


class HypothesisLedgerLens(AnalystLens):
    """Manages the hypothesis ledger based on new evidence.

    For existing hypotheses: checks if new evidence confirms, weakens,
    or falsifies them. For new events: generates candidate hypotheses.
    """

    lens_name = "hypothesis_ledger"
    lens_version = "0.1.0"
    event_classes_supported = ("*",)
    can_modify_existing = True  # Can update existing hypotheses

    def analyze(
        self, packet: AnalysisPacket, config: dict[str, Any] | None = None
    ) -> ModuleDelta:
        config = config or {}
        horizon = config.get("default_horizon_days", 20)
        checkpoint_horizons = tuple(
            int(value)
            for value in config.get("checkpoint_horizons", OUTCOME_HORIZONS)
        )

        new_hypotheses: list[HypothesisLedgerEntry] = []

        # 1. Update existing hypotheses based on new evidence
        existing_updates = self._update_existing_hypotheses(packet)

        # 2. Generate new hypotheses from events
        generated = self._generate_hypotheses(
            packet,
            horizon,
            checkpoint_horizons=checkpoint_horizons,
        )
        new_hypotheses.extend(generated)

        # Combine updated existing + new
        all_hypotheses = list(existing_updates) + new_hypotheses

        return ModuleDelta(
            module_name=self.lens_name,
            module_version=self.lens_version,
            hypotheses_added=new_hypotheses,
            fields_added=["hypotheses"],
            confidence=self._overall_confidence(all_hypotheses),
            reason_for_change=(
                f"Generated {len(new_hypotheses)} new hypotheses, "
                f"updated {len(existing_updates)} existing."
            ),
        )

    def _update_existing_hypotheses(
        self, packet: AnalysisPacket
    ) -> list[HypothesisLedgerEntry]:
        """Check if new evidence confirms, weakens, or falsifies existing hypotheses."""
        updated: list[HypothesisLedgerEntry] = []

        # Gather new evidence text
        new_evidence_text = self._collect_evidence_text(packet)

        for hypothesis in packet.hypotheses:
            if hypothesis.status in (HypothesisStatus.FALSIFIED, HypothesisStatus.CONFIRMED):
                continue  # Already resolved

            status = self._evaluate_hypothesis(hypothesis, new_evidence_text)
            if status != hypothesis.status:
                hypothesis.status = status
                updated.append(hypothesis)

        return updated

    def _evaluate_hypothesis(
        self,
        hypothesis: HypothesisLedgerEntry,
        evidence_text: str,
    ) -> HypothesisStatus:
        """Evaluate whether evidence supports or contradicts a hypothesis."""
        lower = evidence_text.lower()

        # Check invalidation signals
        for signal in hypothesis.invalidation_signals:
            signal_lower = signal.lower()
            # Simple keyword overlap check
            signal_words = set(signal_lower.split())
            evidence_words = set(lower.split())
            overlap = len(signal_words & evidence_words)
            if overlap >= 3:
                return HypothesisStatus.FALSIFIED

        # Check contradicting evidence
        contradicting_hits = 0
        for item_id in hypothesis.contradicting_evidence_ids:
            if item_id in lower:
                contradicting_hits += 1

        # Check supporting evidence
        supporting_hits = 0
        for item_id in hypothesis.supporting_evidence_ids:
            if item_id in lower:
                supporting_hits += 1

        # Evaluate
        if contradicting_hits >= 2:
            return HypothesisStatus.WEAKENED
        if supporting_hits >= 2 and contradicting_hits == 0:
            return HypothesisStatus.CONFIRMED
        return hypothesis.status

    def _generate_hypotheses(
        self,
        packet: AnalysisPacket,
        horizon_days: int,
        *,
        checkpoint_horizons: tuple[int, ...],
    ) -> list[HypothesisLedgerEntry]:
        """Generate new hypotheses from classified events."""
        hypotheses: list[HypothesisLedgerEntry] = []

        events = packet.classified_events or packet.entity_links

        # Generate at most one candidate per canonical event class.
        seen_classes: set[str] = set()
        for event in events:
            if not isinstance(event, dict):
                continue
            source_event_class = str(event.get("event_class", "")).strip()
            event_class = HYPOTHESIS_EVENT_CLASS_ALIASES.get(
                source_event_class, source_event_class
            )
            if event_class in seen_classes or event_class not in HYPOTHESIS_TEMPLATES:
                continue
            seen_classes.add(event_class)

            templates = HYPOTHESIS_TEMPLATES[event_class]
            for tmpl in templates:
                hypothesis_text = tmpl["template"].format(
                    horizon=f"{horizon_days} days",
                )
                entry = HypothesisLedgerEntry(
                    as_of=packet.as_of_date,
                    hypothesis=hypothesis_text,
                    confidence=0.5,
                    supporting_evidence_ids=[
                        event.get("evidence_id") or event.get("event_id", "")
                    ],
                    invalidation_signals=list(tmpl["invalidation_signals"]),
                    expected_observations=list(tmpl["expected_observations"]),
                    horizons_to_check=checkpoint_horizons,
                    status=HypothesisStatus.OPEN,
                    calibration_note=(
                        "Rule-generated candidate hypothesis; confidence is a "
                        "review heuristic, not a calibrated probability."
                    ),
                )
                hypotheses.append(entry)

        return hypotheses

    def _collect_evidence_text(self, packet: AnalysisPacket) -> str:
        parts: list[str] = []
        events = packet.classified_events or [*packet.entity_links, *packet.event_records]
        for event in events:
            if isinstance(event, dict):
                for key in ("text", "text_preview", "title", "summary"):
                    val = event.get(key)
                    if val:
                        parts.append(str(val))
        return " ".join(parts)

    def _overall_confidence(self, hypotheses: list[HypothesisLedgerEntry]) -> float:
        if not hypotheses:
            return 0.3
        open_count = sum(1 for h in hypotheses if h.status == HypothesisStatus.OPEN)
        confirmed = sum(1 for h in hypotheses if h.status == HypothesisStatus.CONFIRMED)
        total = len(hypotheses)
        # Higher confidence when more hypotheses are confirmed
        return 0.3 + (confirmed / max(total, 1)) * 0.3 + (open_count > 0) * 0.1


__all__ = ["HypothesisLedgerLens", "HYPOTHESIS_TEMPLATES"]
