from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from dean_os.unknown_graph import get_domain_unknowns
from dean_os.schemas import (
    AnalyticalReport,
    BaseAgentReport,
    ConsensusDecision,
    PipelineReport,
)

# ── Sector State ──────────────────────────────────────────────────────────


class SectorState(BaseModel):
    """State of one economic sector at a point in time."""

    sector_id: str
    sector_label: str
    stance: str
    confidence: float
    direction: str | None = None
    signal_strength: float | None = None
    data_quality: float = 0.0
    reasons: list[str] = []
    risks: list[str] = []
    evidence_count: int = 0
    ticker_candidates: int = 0
    recommendation: str | None = None
    updated_at: str | None = None


# ── Global State ──────────────────────────────────────────────────────────


class GlobalState(BaseModel):
    """Global macro/regime/prediction context at one point in time."""

    regime: str | None = None
    regime_confidence: float | None = None
    macro_stance: str | None = None
    macro_confidence: float | None = None
    prediction_confidence: float | None = None
    context_synthesis_status: str | None = None
    decision: str | None = None
    decision_confidence: float | None = None
    decision_score: float | None = None


# ── Unknown / Evidence Gap ────────────────────────────────────────────────


class UnknownItem(BaseModel):
    """Something the system knows it doesn't know."""

    description: str
    domain: str | None = None
    source_type: str = "evidence_gap"
    priority: str = "medium"  # low, medium, high
    can_fix_with_collector: bool = False
    requires_human_review: bool = False


# ── World State Snapshot ──────────────────────────────────────────────────


class WorldStateSnapshot(BaseModel):
    """Complete world state at one point in time.

    Built from all agent reports after an orchestrator run.
    Serves as the structured 'morning briefing' foundation.
    """

    snapshot_id: str
    timestamp: str
    as_of: str
    global_state: GlobalState
    sectors: dict[str, SectorState] = {}
    unknowns: list[UnknownItem] = []
    report_count: int = 0
    total_agents: int = 0

    def summary(self) -> str:
        lines: list[str] = []
        lines.append(f"World State @ {self.as_of}")
        lines.append(f"  Decision: {self.global_state.decision} (confidence={self.global_state.decision_confidence})")
        lines.append(f"  Regime: {self.global_state.regime} ({self.global_state.regime_confidence})")
        lines.append(f"  Sectors: {len(self.sectors)}")
        for sid, ss in sorted(self.sectors.items()):
            lines.append(f"    {ss.sector_label}: {ss.stance} (conf={ss.confidence:.2f})")
        if self.unknowns:
            lines.append(f"  Unknowns: {len(self.unknowns)}")
            for u in self.unknowns[:5]:
                lines.append(f"    [{u.priority}] {u.description}")
            if len(self.unknowns) > 5:
                lines.append(f"    ... and {len(self.unknowns) - 5} more")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


# ── Builder ───────────────────────────────────────────────────────────────


class WorldStateBuilder:
    """Builds a WorldStateSnapshot from orchestrator outputs."""

    DOMAIN_SECTOR_MAP: dict[str, str] = {
        "semiconductor_ai_infrastructure": "Semiconductors & AI Infrastructure",
        "energy": "Energy",
        "macro_policy": "Macro / Policy",
        "agriculture": "Agriculture & Soft Commodities",
        "logistics": "Logistics & Supply Chain",
        "real_estate": "Real Estate & REITs",
    }

    def build(
        self,
        reports: list[BaseAgentReport],
        decision: ConsensusDecision | None = None,
        *,
        as_of: str | None = None,
        total_agents: int = 0,
    ) -> WorldStateSnapshot:
        snapshot_id = f"ws_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        timestamp = datetime.now(UTC).isoformat()
        as_of = as_of or timestamp

        global_state = GlobalState()
        sectors: dict[str, SectorState] = {}
        unknowns: list[UnknownItem] = []

        for report in reports:
            self._ingest_report(report, global_state, sectors, unknowns)

        # Seed domain-specific unknowns from UnknownGraph
        for sector_id in sectors:
            domain_graph = get_domain_unknowns(sector_id)
            for entry in domain_graph.get_unresolved()[:3]:
                unknowns.append(
                    UnknownItem(
                        description=entry.description,
                        domain=sector_id,
                        source_type=entry.category,
                        priority=entry.priority,
                        can_fix_with_collector=entry.can_fix_with_collector,
                        requires_human_review=entry.requires_human_review,
                    )
                )

        if decision:
            global_state.decision = decision.decision
            global_state.decision_confidence = decision.confidence
            global_state.decision_score = decision.final_score

        return WorldStateSnapshot(
            snapshot_id=snapshot_id,
            timestamp=timestamp,
            as_of=as_of,
            global_state=global_state,
            sectors=sectors,
            unknowns=unknowns,
            report_count=len(reports),
            total_agents=total_agents,
        )

    def _ingest_report(
        self,
        report: BaseAgentReport,
        global_state: GlobalState,
        sectors: dict[str, SectorState],
        unknowns: list[UnknownItem],
    ) -> None:
        agent_name = report.agent_name
        verdict = report.verdict
        confidence = report.confidence

        if agent_name == "regime":
            global_state.regime = str(verdict)
            global_state.regime_confidence = confidence
            return

        if agent_name == "context_synthesis":
            global_state.context_synthesis_status = str(verdict)
            return

        if agent_name == "macro_analyst" or agent_name == "macro_policy_analyst":
            global_state.macro_stance = str(verdict)
            global_state.macro_confidence = confidence
            return

        sector_id = self._sector_id_from_agent(agent_name)
        if sector_id and sector_id not in sectors:
            sector_label = self.DOMAIN_SECTOR_MAP.get(sector_id, sector_id)
            metrics = getattr(report, "metrics_snapshot", {}) if isinstance(report, PipelineReport) else {}
            evidence_count = metrics.get("evidence_count", 0)
            ticker_candidates = metrics.get("ticker_candidates", 0)
            recommendation = metrics.get("recommendation")

            sectors[sector_id] = SectorState(
                sector_id=sector_id,
                sector_label=sector_label,
                stance=str(verdict),
                confidence=confidence,
                direction=metrics.get("stance") or verdict,
                signal_strength=getattr(report, "signal_strength", None),
                data_quality=getattr(report, "data_quality_score", 0.0),
                reasons=report.reasons[:3] if report.reasons else [],
                risks=report.risks[:3] if report.risks else [],
                evidence_count=evidence_count,
                ticker_candidates=ticker_candidates,
                recommendation=recommendation,
                updated_at=report.timestamp,
            )

        if hasattr(report, "blind_spots") and report.blind_spots:
            for spot in report.blind_spots[:2]:
                unknowns.append(
                    UnknownItem(
                        description=spot,
                        domain=sector_id,
                        source_type="blind_spot",
                        priority="medium",
                    )
                )

    @staticmethod
    def _sector_id_from_agent(agent_name: str) -> str | None:
        mapping = {
            "semiconductor_analyst": "semiconductor_ai_infrastructure",
            "energy_analyst": "energy",
            "macro_analyst": "macro_policy",
            "agriculture_analyst": "agriculture",
            "logistics_analyst": "logistics",
            "real_estate_analyst": "real_estate",
        }
        return mapping.get(agent_name)


__all__ = [
    "GlobalState",
    "SectorState",
    "UnknownItem",
    "WorldStateBuilder",
    "WorldStateSnapshot",
]
