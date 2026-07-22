"""EvidenceGapLens — prioritizes missing evidence by scenario impact.

This lens identifies what evidence is missing and ranks gaps by how
much they would change scenario probabilities. It produces
``evidence_gaps`` entries on the packet.

From design notes §7 §10: "Prioritize missing evidence that would
change scenario probabilities, not merely evidence that sounds interesting."

Deterministic priority scoring. No LLM, no network.
"""
from __future__ import annotations

from typing import Any

from dean_os.analyst_core.lens_contract import AnalysisPacket, AnalystLens, ModuleDelta
from dean_os.analyst_core.schemas import EvidenceGap, Priority
from dean_os.utils import sha256_json

# ──────────────────────────────────────────────────────────────────────────────
# Evidence gap templates per event class (from design notes §7 §10)
# Maps event classes to likely evidence gaps.
# ──────────────────────────────────────────────────────────────────────────────

EVIDENCE_GAP_TEMPLATES: dict[str, list[dict[str, Any]]] = {
    "demand_driver": [
        {
            "description": "Actual order backlog data vs. narrative claims",
            "importance": Priority.HIGH,
            "source_type": "company_filing",
        },
        {
            "description": "Hyperscaler capex guidance vs. analyst estimates",
            "importance": Priority.HIGH,
            "source_type": "earnings_call",
        },
        {
            "description": "Enterprise AI ROI metrics from early adopters",
            "importance": Priority.MEDIUM,
            "source_type": "industry_report",
        },
    ],
    "supply_disruption": [
        {
            "description": "Actual production capacity and utilization rates",
            "importance": Priority.HIGH,
            "source_type": "industry_data",
        },
        {
            "description": "Lead time data from multiple suppliers",
            "importance": Priority.HIGH,
            "source_type": "company_data",
        },
        {
            "description": "Inventory levels across supply chain",
            "importance": Priority.MEDIUM,
            "source_type": "industry_report",
        },
    ],
    "oil_shock": [
        {
            "description": "Actual production cut volume and timeline",
            "importance": Priority.HIGH,
            "source_type": "official_statement",
        },
        {
            "description": "Strategic petroleum reserve levels and policy",
            "importance": Priority.HIGH,
            "source_type": "government_data",
        },
        {
            "description": "Demand elasticity from consuming nations",
            "importance": Priority.MEDIUM,
            "source_type": "economic_data",
        },
    ],
    "tariff": [
        {
            "description": "Exact tariff rates and scope of application",
            "importance": Priority.HIGH,
            "source_type": "government_regulation",
        },
        {
            "description": "Supply chain diversification timelines and costs",
            "importance": Priority.HIGH,
            "source_type": "company_disclosure",
        },
        {
            "description": "Retaliatory tariff risk from trade partners",
            "importance": Priority.MEDIUM,
            "source_type": "geopolitical_analysis",
        },
    ],
    "sanctions_change": [
        {
            "description": "Enforcement mechanism and compliance requirements",
            "importance": Priority.HIGH,
            "source_type": "government_regulation",
        },
        {
            "description": "Alternative supply channel availability",
            "importance": Priority.HIGH,
            "source_type": "industry_intelligence",
        },
    ],
    "central_bank_decision": [
        {
            "description": "Forward guidance language and dot plot",
            "importance": Priority.HIGH,
            "source_type": "central_bank_communication",
        },
        {
            "description": "Market-implied rate path from futures",
            "importance": Priority.MEDIUM,
            "source_type": "market_data",
        },
    ],
    "memory_supply_constraint": [
        {
            "description": "HBM contract pricing and allocation data",
            "importance": Priority.HIGH,
            "source_type": "industry_data",
        },
        {
            "description": "DRAM spot vs. contract price divergence",
            "importance": Priority.MEDIUM,
            "source_type": "market_data",
        },
    ],
    "capex_signal": [
        {
            "description": "Capex breakdown: maintenance vs. growth",
            "importance": Priority.HIGH,
            "source_type": "company_filing",
        },
        {
            "description": "Equipment order data from suppliers",
            "importance": Priority.MEDIUM,
            "source_type": "industry_data",
        },
    ],
    # ── Full-Economy Gaps ──────────────────────────────────────────────────
    "climate_disaster": [
        {
            "description": "Exact downtime estimate for affected energy/industrial facilities",
            "importance": Priority.HIGH,
            "source_type": "company_disclosure",
        },
        {
            "description": "Insurance claims vs reserves ratio",
            "importance": Priority.MEDIUM,
            "source_type": "financial_data",
        },
    ],
    "trade_route_disruption": [
        {
            "description": "Volume of cargo rerouted vs delayed in transit",
            "importance": Priority.HIGH,
            "source_type": "logistics_data",
        },
        {
            "description": "Spot vs contract freight rate spread",
            "importance": Priority.MEDIUM,
            "source_type": "market_data",
        },
    ],
    "political_transition": [
        {
            "description": "Cabinet appointments for key economic ministries",
            "importance": Priority.HIGH,
            "source_type": "government_announcement",
        },
        {
            "description": "Status of pending regulatory approvals under new administration",
            "importance": Priority.HIGH,
            "source_type": "policy_analysis",
        },
    ],
    "debt_crisis": [
        {
            "description": "Bank exposure to downgraded sovereign debt",
            "importance": Priority.HIGH,
            "source_type": "company_filing",
        },
        {
            "description": "Bailout terms and conditionality",
            "importance": Priority.HIGH,
            "source_type": "government_announcement",
        },
    ],
    "pandemic_health_shock": [
        {
            "description": "Mobility index data for key industrial regions",
            "importance": Priority.HIGH,
            "source_type": "alternative_data",
        },
        {
            "description": "Fiscal support package size vs expected GDP hit",
            "importance": Priority.MEDIUM,
            "source_type": "policy_analysis",
        },
    ],
}


class EvidenceGapLens(AnalystLens):
    """Prioritizes evidence gaps by their impact on scenario probabilities.

    Reads existing evidence gaps from the packet, generates new gap
    candidates from events, and produces a prioritized list.
    """

    lens_name = "evidence_gap"
    lens_version = "0.1.0"
    event_classes_supported = ("*",)
    can_modify_existing = True  # Can add to existing gaps

    def analyze(
        self, packet: AnalysisPacket, config: dict[str, Any] | None = None
    ) -> ModuleDelta:
        existing_gap_descriptions = {
            gap.description for gap in packet.evidence_gaps
        }

        new_gaps: list[EvidenceGap] = []

        events = packet.classified_events or packet.entity_links

        # Generate gaps from classified events
        for event in events:
            if not isinstance(event, dict):
                continue
            event_class = str(event.get("event_class", "")).strip()
            if event_class not in EVIDENCE_GAP_TEMPLATES:
                continue

            templates = EVIDENCE_GAP_TEMPLATES[event_class]
            for tmpl in templates:
                if tmpl["description"] not in existing_gap_descriptions:
                    gap = EvidenceGap(
                        description=tmpl["description"],
                        importance_to_scenario_probability=tmpl["importance"],
                        expected_source_type=tmpl["source_type"],
                        current_status="missing",
                        priority=tmpl["importance"],
                    )
                    new_gaps.append(gap)
                    existing_gap_descriptions.add(tmpl["description"])

        # Check for gaps implied by scenario graph
        if packet.scenario_graph:
            graph_gaps = self._gaps_from_scenario_graph(packet.scenario_graph)
            for gap in graph_gaps:
                if gap.description not in existing_gap_descriptions:
                    new_gaps.append(gap)
                    existing_gap_descriptions.add(gap.description)

        # Check for gaps implied by hypotheses
        for hypothesis in packet.hypotheses:
            if not hypothesis.expected_observations:
                continue
            desc = (
                f"Observation needed to test hypothesis: "
                f"{hypothesis.hypothesis[:80]}"
            )
            if desc not in existing_gap_descriptions:
                gap = EvidenceGap(
                    description=desc,
                    importance_to_scenario_probability=Priority.HIGH
                    if hypothesis.confidence > 0.6
                    else Priority.MEDIUM,
                    expected_source_type="market_or_company_data",
                    current_status="missing",
                    priority=Priority.HIGH
                    if hypothesis.confidence > 0.6
                    else Priority.MEDIUM,
                )
                new_gaps.append(gap)

        review_notes: list[str] = []
        high_priority = sum(
            1 for g in new_gaps if g.priority == Priority.HIGH
        )
        if high_priority > 0:
            review_notes.append(
                f"evidence_gap: {high_priority} HIGH priority gaps identified"
            )

        for gap in new_gaps:
            gap.gap_id = "gap_" + sha256_json(
                {
                    "description": gap.description,
                    "expected_source_type": gap.expected_source_type,
                    "as_of": packet.as_of_date,
                }
            )[:24]

        return ModuleDelta(
            module_name=self.lens_name,
            module_version=self.lens_version,
            as_of=packet.as_of_date,
            evidence_gaps_added=new_gaps,
            fields_added=["evidence_gaps"],
            confidence=self._overall_confidence(new_gaps, len(packet.evidence_gaps)),
            reason_for_change=(
                f"Added {len(new_gaps)} evidence gaps "
                f"({high_priority} high priority)."
            ),
            review_notes_added=review_notes,
        )

    def _gaps_from_scenario_graph(
        self, graph: Any
    ) -> list[EvidenceGap]:
        """Extract evidence gaps from scenario graph uncertainty notes."""
        gaps: list[EvidenceGap] = []
        if not hasattr(graph, "nodes"):
            return gaps

        for node in graph.nodes:
            if hasattr(node, "uncertainty_notes") and node.uncertainty_notes:
                gaps.append(
                    EvidenceGap(
                        description=f"Uncertainty in scenario node '{node.label}': {node.uncertainty_notes}",
                        importance_to_scenario_probability=Priority.MEDIUM,
                        expected_source_type="scenario_analysis",
                        current_status="identified",
                        priority=Priority.MEDIUM,
                    )
                )

        return gaps

    def _overall_confidence(
        self, new_gaps: list[EvidenceGap], existing_count: int
    ) -> float:
        total = len(new_gaps) + existing_count
        if total == 0:
            return 0.3
        high = sum(1 for g in new_gaps if g.priority == Priority.HIGH)
        return 0.3 + min(high / max(total, 1), 1.0) * 0.3


__all__ = ["EvidenceGapLens", "EVIDENCE_GAP_TEMPLATES"]
