from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from dean_os.draft.dean_os_agent_system_v7.dean_os.collector_routing import CollectorRoute, DomainCollectorRouter


class EvidenceGapTask(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str = Field(default_factory=lambda: f"evidence_gap_{uuid4().hex}")
    domain_id: str
    coverage_id: str
    priority: Literal["critical", "high", "medium", "low"]
    reason: str
    target_evidence_types: list[str] = Field(default_factory=list)
    collector_routes: list[CollectorRoute] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    review_only: bool = True
    can_execute_network: bool = False
    can_write_learning_memory: bool = False


class EvidenceGapPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    domain_id: str
    run_id: str
    tasks: list[EvidenceGapTask] = Field(default_factory=list)
    unresolved_coverage_ids: list[str] = Field(default_factory=list)
    weak_coverage_ids: list[str] = Field(default_factory=list)
    blocked: bool = False


class EvidenceGapPlanner:
    def __init__(self, domain_id: str):
        self.domain_id = domain_id
        self.router = DomainCollectorRouter(domain_id)

    def build(self, *, briefing: Any, evidence_records: list[dict[str, Any]]) -> EvidenceGapPlan:
        data = briefing.model_dump(mode="json") if hasattr(briefing, "model_dump") else dict(briefing)
        records_by_lane: dict[str, list[dict[str, Any]]] = {}
        for record in evidence_records:
            for lane in record.get("evidence_lanes", []) or []:
                records_by_lane.setdefault(str(lane), []).append(record)

        tasks: list[EvidenceGapTask] = []
        unresolved: list[str] = []
        weak: list[str] = []
        for item in data.get("mandatory_coverage_gate", []) or []:
            coverage_id = str(item.get("coverage_id"))
            aliases = [coverage_id] + [str(value) for value in item.get("aliases", []) or []]
            lane_records = [record for alias in aliases for record in records_by_lane.get(alias, [])]
            credible = [record for record in lane_records if float(record.get("credibility_score", 0.0)) >= 0.70]
            independent_sources = {str(record.get("source_name")) for record in credible}
            status = item.get("status")
            if status in {"no_credible_material_update", "evidence_gap"}:
                unresolved.append(coverage_id)
                tasks.append(self._task(
                    coverage_id,
                    priority="high" if status == "evidence_gap" else "medium",
                    reason=str(item.get("evidence_gap") or item.get("conclusion") or "coverage is unresolved"),
                ))
            elif lane_records and not credible:
                weak.append(coverage_id)
                tasks.append(self._task(
                    coverage_id,
                    priority="high",
                    reason="current lane is supported only by weak or quarantined sources",
                ))
            elif credible and len(independent_sources) < 2:
                weak.append(coverage_id)
                tasks.append(self._task(
                    coverage_id,
                    priority="medium",
                    reason="material lane lacks independent corroboration",
                ))

        return EvidenceGapPlan(
            domain_id=self.domain_id,
            run_id=str(data.get("run_id") or "unknown_run"),
            tasks=tasks,
            unresolved_coverage_ids=sorted(set(unresolved)),
            weak_coverage_ids=sorted(set(weak)),
            blocked=False,
        )

    def _task(self, coverage_id: str, *, priority: str, reason: str) -> EvidenceGapTask:
        return EvidenceGapTask(
            domain_id=self.domain_id,
            coverage_id=coverage_id,
            priority=priority,
            reason=reason,
            target_evidence_types=[coverage_id],
            collector_routes=self.router.routes_for(coverage_id),
            acceptance_criteria=[
                "At least one point-in-time valid source with credibility_score >= 0.70.",
                "Material claims must retain source anchors and availability timestamps.",
                "Prefer two independent sources unless a primary official source is sufficient.",
            ],
        )


__all__ = ["EvidenceGapTask", "EvidenceGapPlan", "EvidenceGapPlanner"]
