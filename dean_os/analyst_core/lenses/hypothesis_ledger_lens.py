"""HypothesisLedgerLens — manages the lifecycle of analyst hypotheses.

This lens creates hypotheses and proposes review actions based on new evidence.
It never confirms, weakens, or falsifies a hypothesis by itself.

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
from dean_os.utils import sha256_json

# ──────────────────────────────────────────────────────────────────────────────
# Hypothesis generation rules (from design notes §6.13)
# Maps event classes to candidate hypothesis templates.
# ──────────────────────────────────────────────────────────────────────────────

HYPOTHESIS_TEMPLATES: dict[str, list[dict[str, Any]]] = {
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
    # ── Full-Economy event hypotheses ─────────────────────────────────────────────────
    "climate_disaster": [
        {
            "template": "Physical damage will disrupt regional energy and supply chains over {horizon}",
            "invalidation_signals": [
                "infrastructure restored ahead of schedule",
                "impact contained to single facility",
                "insurance and reserves absorb full cost",
            ],
            "expected_observations": [
                "regional power price spike",
                "freight rate increase on affected routes",
                "production halt announcements",
            ],
        },
    ],
    "trade_route_disruption": [
        {
            "template": "Shipping lane disruption will raise global freight costs over {horizon}",
            "invalidation_signals": [
                "alternative route reopened at comparable cost",
                "diplomatic resolution restores passage",
                "inventory buffers absorb delay",
            ],
            "expected_observations": [
                "Baltic Dry Index or container rate spike",
                "extended lead times from affected regions",
                "rerouting announcements from major carriers",
            ],
        },
    ],
    "political_transition": [
        {
            "template": "Policy uncertainty from transition will defer investment over {horizon}",
            "invalidation_signals": [
                "new government confirms policy continuity",
                "markets normalize quickly after election",
                "constitutional constraints limit policy change",
            ],
            "expected_observations": [
                "fx volatility or capital outflow",
                "investment deferral announcements",
                "regulatory review pause",
            ],
        },
    ],
    "debt_crisis": [
        {
            "template": "Credit deterioration will tighten financial conditions over {horizon}",
            "invalidation_signals": [
                "IMF or ECB backstop approved",
                "debt restructuring completed",
                "contagion contained to single country",
            ],
            "expected_observations": [
                "sovereign spread widening",
                "banking sector selling pressure",
                "global risk-off rotation",
            ],
        },
    ],
    "pandemic_health_shock": [
        {
            "template": "Mobility restrictions will compress consumer demand over {horizon}",
            "invalidation_signals": [
                "variant proves less severe than feared",
                "vaccine rollout contains spread quickly",
                "governments resist lockdown measures",
            ],
            "expected_observations": [
                "mobility index decline",
                "retail sales contraction",
                "supply chain disruption reports",
            ],
        },
    ],
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
        review_proposals = self._propose_existing_hypothesis_reviews(packet)

        # 2. Generate new hypotheses from events
        generated = self._generate_hypotheses(
            packet,
            horizon,
            checkpoint_horizons=checkpoint_horizons,
        )
        new_hypotheses.extend(generated)

        # Combine updated existing + new
        all_hypotheses = list(packet.hypotheses) + new_hypotheses

        return ModuleDelta(
            module_name=self.lens_name,
            module_version=self.lens_version,
            as_of=packet.as_of_date,
            hypotheses_added=new_hypotheses,
            hypothesis_review_proposals_added=review_proposals,
            fields_added=[
                field
                for field, values in (
                    ("hypotheses", new_hypotheses),
                    ("hypothesis_review_proposals", review_proposals),
                )
                if values
            ],
            evidence_ids=sorted(
                {
                    evidence_id
                    for hypothesis in new_hypotheses
                    for evidence_id in hypothesis.trigger_evidence_ids
                }.union(
                    evidence_id
                    for proposal in review_proposals
                    for evidence_id in proposal.get("evidence_ids", [])
                )
            ),
            confidence=self._overall_confidence(all_hypotheses),
            reason_for_change=(
                f"Generated {len(new_hypotheses)} new hypotheses, "
                f"proposed {len(review_proposals)} review actions; "
                "no hypothesis status was changed automatically."
            ),
        )

    def _propose_existing_hypothesis_reviews(
        self, packet: AnalysisPacket
    ) -> list[dict[str, Any]]:
        """Propose status reviews without mutating the canonical hypothesis."""
        proposals: list[dict[str, Any]] = []

        for hypothesis in packet.hypotheses:
            if hypothesis.status in (HypothesisStatus.FALSIFIED, HypothesisStatus.CONFIRMED):
                continue
            proposal = self._evaluate_hypothesis(hypothesis, packet)
            if proposal is not None:
                proposals.append(proposal)

        return proposals

    def _evaluate_hypothesis(
        self,
        hypothesis: HypothesisLedgerEntry,
        packet: AnalysisPacket,
    ) -> dict[str, Any] | None:
        """Find a candidate support/contradiction match for human/outcome review."""
        best: dict[str, Any] | None = None
        events = packet.classified_events or [*packet.event_records, *packet.entity_links]
        signal_groups = (
            ("candidate_contradiction", hypothesis.invalidation_signals, HypothesisStatus.WEAKENED),
            ("candidate_support", hypothesis.expected_observations, hypothesis.status),
        )
        for event in events:
            if not isinstance(event, dict):
                continue
            text = " ".join(
                str(event.get(key) or "")
                for key in ("text", "text_preview", "title", "summary")
            ).lower()
            evidence_words = set(text.split())
            evidence_id = str(
                event.get("evidence_id") or event.get("event_id") or ""
            ).strip()
            if not evidence_words or not evidence_id:
                continue
            for proposal_type, signals, suggested_status in signal_groups:
                for signal in signals:
                    signal_words = set(signal.lower().split())
                    if not signal_words:
                        continue
                    score = len(signal_words & evidence_words) / len(signal_words)
                    threshold = 0.6 if proposal_type == "candidate_contradiction" else 0.5
                    if score < threshold:
                        continue
                    candidate = {
                        "proposal_id": "hypothesis_review_" + sha256_json(
                            {
                                "hypothesis_id": hypothesis.hypothesis_id,
                                "proposal_type": proposal_type,
                                "evidence_id": evidence_id,
                                "signal": signal,
                                "as_of": packet.as_of_date,
                            }
                        )[:24],
                        "hypothesis_id": hypothesis.hypothesis_id,
                        "current_status": hypothesis.status.value,
                        "suggested_status": suggested_status.value,
                        "proposal_type": proposal_type,
                        "matched_signal": signal,
                        "match_score": round(score, 6),
                        "evidence_ids": [evidence_id],
                        "as_of": packet.as_of_date,
                        "requires_manual_review": True,
                        "requires_outcome_evidence": True,
                        "status_changed": False,
                    }
                    if best is None or candidate["match_score"] > best["match_score"]:
                        best = candidate
        return best

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
            event_class = str(event.get("event_class", "")).strip()
            if event_class in seen_classes or event_class not in HYPOTHESIS_TEMPLATES:
                continue
            seen_classes.add(event_class)

            templates = HYPOTHESIS_TEMPLATES[event_class]
            for tmpl in templates:
                hypothesis_text = tmpl["template"].format(
                    horizon=f"{horizon_days} days",
                )
                entry = HypothesisLedgerEntry(
                    hypothesis_id="hypothesis_" + sha256_json(
                        {
                            "hypothesis": hypothesis_text,
                            "event_class": event_class,
                            "trigger_evidence_id": (
                                event.get("evidence_id")
                                or event.get("event_id", "")
                            ),
                            "as_of": packet.as_of_date,
                        }
                    )[:24],
                    as_of=packet.as_of_date,
                    hypothesis=hypothesis_text,
                    confidence=0.5,
                    trigger_evidence_ids=[
                        event.get("evidence_id") or event.get("event_id", "")
                    ],
                    supporting_evidence_ids=[],
                    invalidation_signals=list(tmpl["invalidation_signals"]),
                    expected_observations=list(tmpl["expected_observations"]),
                    horizons_to_check=checkpoint_horizons,
                    status=HypothesisStatus.OPEN,
                    calibration_note=(
                        "Rule-generated candidate hypothesis. The linked event "
                        "is trigger evidence, not supporting proof; confidence "
                        "is a review heuristic, not a calibrated probability."
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
