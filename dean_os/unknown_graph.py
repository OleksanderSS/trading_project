from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ValueOfInformationAssessment(BaseModel):
    """Ordinal collector triage, not monetary expected value or probability."""

    status: Literal["unassessed", "draft", "validated", "rejected"] = "unassessed"
    uncertainty_type: Literal["epistemic", "aleatoric", "mixed", "unknown"] = "unknown"
    scenario_change_potential: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_change_potential: float | None = Field(default=None, ge=0.0, le=1.0)
    wrong_conclusion_blocking_value: float | None = Field(default=None, ge=0.0, le=1.0)
    decision_relevance: float | None = Field(default=None, ge=0.0, le=1.0)
    collection_feasibility: float | None = Field(default=None, ge=0.0, le=1.0)
    normalized_collection_cost: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_basis: list[str] = Field(default_factory=list)
    assessor: str | None = None
    assessed_at: str | None = None
    triage_score: float | None = Field(default=None, ge=0.0, le=1.0)
    limitations: list[str] = Field(default_factory=list)

    def calculate(self) -> "ValueOfInformationAssessment":
        components = (
            self.scenario_change_potential,
            self.confidence_change_potential,
            self.wrong_conclusion_blocking_value,
            self.decision_relevance,
            self.collection_feasibility,
            self.normalized_collection_cost,
        )
        if self.status != "validated" or any(value is None for value in components):
            self.triage_score = None
            return self
        if not self.evidence_basis or not self.assessor or not self.assessed_at:
            self.triage_score = None
            return self
        decision_value = (
            0.35 * self.scenario_change_potential
            + 0.20 * self.confidence_change_potential
            + 0.30 * self.wrong_conclusion_blocking_value
            + 0.15 * self.decision_relevance
        )
        effort_factor = self.collection_feasibility * (1.0 - 0.5 * self.normalized_collection_cost)
        self.triage_score = round(decision_value * effort_factor, 4)
        return self


class UnknownEntry(BaseModel):
    """Something the system explicitly tracks as unknown."""

    id: str = ""
    description: str
    domain: str = "general"
    category: str = "evidence_gap"  # missing_data, unverified, conflicting, unknown_transmission, unknown_lag, unknown_impact, need_collector, need_review
    priority: str = "medium"  # low, medium, high, critical
    created_at: str = ""
    updated_at: str = ""
    can_fix_with_collector: bool = False
    collector_type: str = ""
    requires_human_review: bool = False
    related_sectors: list[str] = []
    related_tickers: list[str] = []
    attempted_resolution: bool = False
    resolved_at: str | None = None
    resolution_note: str | None = None
    linked_hypothesis_ids: list[str] = Field(default_factory=list)
    linked_scenario_ids: list[str] = Field(default_factory=list)
    voi: ValueOfInformationAssessment = Field(default_factory=ValueOfInformationAssessment)


class UnknownGraph(BaseModel):
    """First-class tracking of system unknowns.

    Each unknown is a structured entry that can be:
    - Tracked across time (created → updated → resolved)
    - Prioritized (low → critical)
    - Actioned (collector assignment, human review)
    - Learned from (resolution patterns)
    """

    domain: str = "general"
    entries: list[UnknownEntry] = Field(default_factory=list)

    def add(
        self,
        description: str,
        *,
        domain: str | None = None,
        category: str = "evidence_gap",
        priority: str = "medium",
        can_fix_with_collector: bool = False,
        collector_type: str = "",
        requires_human_review: bool = False,
        related_sectors: list[str] | None = None,
        related_tickers: list[str] | None = None,
        linked_hypothesis_ids: list[str] | None = None,
        linked_scenario_ids: list[str] | None = None,
        voi: ValueOfInformationAssessment | dict[str, Any] | None = None,
    ) -> UnknownEntry:
        entry = UnknownEntry(
            id=f"unk_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{len(self.entries)}",
            description=description,
            domain=domain or self.domain,
            category=category,
            priority=priority,
            created_at=datetime.now(UTC).isoformat(),
            updated_at=datetime.now(UTC).isoformat(),
            can_fix_with_collector=can_fix_with_collector,
            collector_type=collector_type,
            requires_human_review=requires_human_review,
            related_sectors=related_sectors or [],
            related_tickers=related_tickers or [],
            linked_hypothesis_ids=linked_hypothesis_ids or [],
            linked_scenario_ids=linked_scenario_ids or [],
            voi=(
                voi
                if isinstance(voi, ValueOfInformationAssessment)
                else ValueOfInformationAssessment.model_validate(voi or {})
            ),
        )
        self.entries.append(entry)
        return entry

    def resolve(self, entry_id: str, note: str = "") -> bool:
        for entry in self.entries:
            if entry.id == entry_id and not entry.resolved_at:
                entry.resolved_at = datetime.now(UTC).isoformat()
                entry.resolution_note = note
                entry.attempted_resolution = True
                return True
        return False

    def get_high_priority(self) -> list[UnknownEntry]:
        return [e for e in self.entries if e.priority in ("high", "critical") and not e.resolved_at]

    def get_collector_fixable(self) -> list[UnknownEntry]:
        return [e for e in self.entries if e.can_fix_with_collector and not e.resolved_at]

    def get_unresolved(self) -> list[UnknownEntry]:
        return [e for e in self.entries if not e.resolved_at]

    def assess_value_of_information(
        self,
        entry_id: str,
        assessment: ValueOfInformationAssessment | dict[str, Any],
    ) -> bool:
        value = (
            assessment
            if isinstance(assessment, ValueOfInformationAssessment)
            else ValueOfInformationAssessment.model_validate(assessment)
        ).calculate()
        for entry in self.entries:
            if entry.id == entry_id and not entry.resolved_at:
                entry.voi = value
                entry.updated_at = datetime.now(UTC).isoformat()
                return True
        return False

    def prioritized_collector_backlog(self) -> list[UnknownEntry]:
        """Validated VoI first; unassessed entries remain visible but unscored."""
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        return sorted(
            self.get_collector_fixable(),
            key=lambda entry: (
                entry.voi.triage_score is None,
                -(entry.voi.triage_score or 0.0),
                priority_order.get(entry.priority, 4),
                entry.created_at,
                entry.id,
            ),
        )

    def summary(self) -> str:
        unresolved = self.get_unresolved()
        high = self.get_high_priority()
        fixable = self.get_collector_fixable()
        if not unresolved:
            return f"[{self.domain}] No unknowns tracked"
        return (
            f"[{self.domain}] {len(unresolved)} unknowns "
            f"({len(high)} high priority, {len(fixable)} collector-fixable)"
        )


# ── Default unknown graphs for key domains ────────────────────────────────

DEFAULT_UNKNOWNS: dict[str, list[dict[str, Any]]] = {
    "semiconductor_ai_infrastructure": [
        {"description": "Real-time CoWoS capacity utilization", "category": "missing_data", "can_fix_with_collector": True, "collector_type": "supply_chain", "priority": "high"},
        {"description": "Hyperscaler capex breakdown by AI vs non-AI", "category": "missing_data", "can_fix_with_collector": True, "collector_type": "earnings_analysis", "priority": "high"},
        {"description": "HBM3e qualification status by supplier", "category": "unverified", "requires_human_review": True, "priority": "medium"},
        {"description": "Export control impact on specific Chinese AI chip startups", "category": "unverified", "priority": "medium"},
        {"description": "Data center power grid interconnection queue times by region", "category": "missing_data", "can_fix_with_collector": True, "collector_type": "energy_data", "priority": "low"},
        {"description": "GPU availability lead times for enterprise vs hyperscaler", "category": "unknown_lag", "priority": "high"},
    ],
    "energy": [
        {"description": "Real-time OPEC+ production vs quota compliance", "category": "unverified", "priority": "high", "can_fix_with_collector": True, "collector_type": "industry_data"},
        {"description": "US strategic petroleum reserve fill rate", "category": "missing_data", "priority": "medium"},
        {"description": "Global refinery capacity additions vs closures net balance", "category": "unknown_impact", "priority": "high"},
    ],
    "general": [
        {"description": "Real-time global port congestion index", "category": "missing_data", "can_fix_with_collector": True, "collector_type": "logistics", "priority": "medium"},
        {"description": "Central bank liquidity measures outside Fed (PBoC, ECB, BOJ)", "category": "missing_data", "priority": "high"},
        {"description": "Cross-border capital flow data by region", "category": "missing_data", "priority": "medium"},
    ],
}


def get_domain_unknowns(domain_id: str) -> UnknownGraph:
    """Get default unknown entries for a domain."""
    graph = UnknownGraph(domain=domain_id)
    raw_list = DEFAULT_UNKNOWNS.get(domain_id, []) + DEFAULT_UNKNOWNS.get("general", [])
    for item in raw_list:
        graph.add(**item)
    return graph


__all__ = [
    "UnknownEntry",
    "UnknownGraph",
    "ValueOfInformationAssessment",
    "get_domain_unknowns",
]
