from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


BRIEFING_SCHEMA_VERSION = "dean_daily_briefing_v1"


class CoverageGateItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    coverage_id: str
    label: str
    aliases: list[str] = Field(default_factory=list)
    status: Literal["material_update", "no_credible_material_update", "evidence_gap"]
    evidence_ids: list[str] = Field(default_factory=list)
    conclusion: str
    evidence_gap: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> "CoverageGateItem":
        if self.status == "material_update" and not self.evidence_ids:
            raise ValueError("material_update requires evidence_ids")
        if self.status == "evidence_gap" and not self.evidence_gap:
            raise ValueError("evidence_gap status requires evidence_gap")
        return self


class DailyBriefing(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = BRIEFING_SCHEMA_VERSION
    briefing_id: str = Field(default_factory=lambda: f"briefing_{uuid4().hex}")
    run_id: str
    domain_id: str
    as_of: str
    regime_snapshot: dict[str, Any] = Field(default_factory=dict)
    mandatory_coverage_gate: list[CoverageGateItem]
    top_developments: list[dict[str, Any]] = Field(default_factory=list)
    context_grid: dict[str, Any] = Field(default_factory=dict)
    indicator_state_grid: dict[str, Any] = Field(default_factory=dict)
    scenario_probabilities: dict[str, Any] = Field(default_factory=dict)
    practical_implications: list[str] = Field(default_factory=list)
    risks_and_evidence_gaps: list[str] = Field(default_factory=list)
    replay_journal: dict[str, Any] = Field(default_factory=dict)
    evidence_quality_summary: dict[str, Any] = Field(default_factory=dict)
    review_only: bool = True
    human_review_required: bool = True

    @model_validator(mode="after")
    def _validate_gate(self) -> "DailyBriefing":
        ids = [item.coverage_id for item in self.mandatory_coverage_gate]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("mandatory coverage gate must contain unique coverage items")
        if not self.review_only or not self.human_review_required:
            raise ValueError("v1 briefing must remain review-only")
        return self


class DailyBriefingBuilder:
    def build(
        self,
        *,
        run_result: Any,
        required_coverage: list[dict[str, str]],
        evidence_records: list[dict[str, Any]],
        replay_due: list[dict[str, Any]] | None = None,
    ) -> DailyBriefing:
        result = run_result.model_dump(mode="json") if hasattr(run_result, "model_dump") else dict(run_result)
        evidence_by_lane: dict[str, list[str]] = {}
        weak_evidence_by_lane: dict[str, list[str]] = {}
        credibility_scores: list[float] = []
        for record in evidence_records:
            score = float(record.get("credibility_score", record.get("quality_score", 0.75)))
            credibility_scores.append(score)
            target = evidence_by_lane if score >= 0.70 and record.get("point_in_time_status", "valid") == "valid" else weak_evidence_by_lane
            for lane in record.get("evidence_lanes", []) or []:
                target.setdefault(str(lane), []).append(str(record.get("evidence_id")))
            for sector in record.get("sectors", []) or []:
                target.setdefault(str(sector), []).append(str(record.get("evidence_id")))

        coverage_items: list[CoverageGateItem] = []
        for item in required_coverage:
            coverage_id = str(item["coverage_id"])
            aliases = [coverage_id] + [str(value) for value in item.get("aliases", []) or []]
            evidence_ids = sorted({evidence_id for alias in aliases for evidence_id in evidence_by_lane.get(alias, [])})
            weak_ids = sorted({evidence_id for alias in aliases for evidence_id in weak_evidence_by_lane.get(alias, [])})
            if evidence_ids:
                coverage_items.append(CoverageGateItem(
                    coverage_id=coverage_id,
                    label=str(item.get("label") or coverage_id),
                    aliases=sorted(set(aliases)),
                    status="material_update",
                    evidence_ids=evidence_ids,
                    conclusion=f"Credible evidence was cataloged for {item.get('label') or coverage_id}.",
                ))
            elif weak_ids:
                coverage_items.append(CoverageGateItem(
                    coverage_id=coverage_id,
                    label=str(item.get("label") or coverage_id),
                    aliases=sorted(set(aliases)),
                    status="evidence_gap",
                    evidence_ids=[],
                    conclusion="Only weak, quarantined, or point-in-time-invalid evidence was found.",
                    evidence_gap="Replace or corroborate weak evidence before a material conclusion.",
                ))
            else:
                coverage_items.append(CoverageGateItem(
                    coverage_id=coverage_id,
                    label=str(item.get("label") or coverage_id),
                    aliases=sorted(set(aliases)),
                    status="no_credible_material_update",
                    conclusion="No credible material update found in the current bounded evidence set.",
                ))

        world_model = result.get("world_model_event_learning", {}) or {}
        scenario_graph = world_model.get("scenario_outcome_graph", {}) or {}
        nodes = list(scenario_graph.get("nodes", []) or [])
        nodes.sort(key=lambda node: float(node.get("probability", 0.0)), reverse=True)
        top_developments = []
        for event in list(world_model.get("classified_events", []) or [])[:5]:
            top_developments.append({
                "event_id": event.get("event_id"),
                "event_type": event.get("event_type"),
                "summary": event.get("summary") or event.get("description"),
                "evidence_ids": event.get("evidence_ids", []),
            })

        context_grid = result.get("context_grid", {}) or {}
        regime_snapshot = _extract_regime(context_grid)
        evidence_gaps = [
            str(item.get("description") or item)
            for item in world_model.get("evidence_gaps", []) or []
        ]
        replay_tasks = list(world_model.get("replay_tasks", []) or [])
        return DailyBriefing(
            run_id=str(result.get("run_id") or "unknown_run"),
            domain_id=str(result.get("domain_id") or "unknown_domain"),
            as_of=str((result.get("world_state_snapshot") or {}).get("as_of") or regime_snapshot.get("as_of") or ""),
            regime_snapshot=regime_snapshot,
            mandatory_coverage_gate=coverage_items,
            top_developments=top_developments,
            context_grid=context_grid,
            indicator_state_grid=result.get("indicator_state_grid", {}) or {},
            scenario_probabilities={
                "scenario_graph_id": scenario_graph.get("scenario_graph_id"),
                "nodes": nodes,
                "edges": scenario_graph.get("edges", []),
                "probability_mass": sum(float(node.get("probability", 0.0)) for node in nodes),
            },
            practical_implications=_practical_implications(nodes, result),
            risks_and_evidence_gaps=sorted(set(evidence_gaps + list(result.get("decision", {}).get("risks", []) or []))),
            replay_journal={
                "created_replay_tasks": replay_tasks,
                "due_replay_tasks": list(replay_due or []),
                "historical_analogs": result.get("historical_world_state_analogs", []),
                "learning_write_allowed": False,
            },
            evidence_quality_summary={
                "record_count": len(evidence_records),
                "credible_record_count": sum(1 for value in credibility_scores if value >= 0.70),
                "weak_record_count": sum(1 for value in credibility_scores if value < 0.45),
                "mean_credibility": (sum(credibility_scores) / len(credibility_scores)) if credibility_scores else None,
            },
        )


def _extract_regime(context_grid: dict[str, Any]) -> dict[str, Any]:
    nodes = list(context_grid.get("nodes", []) or [])
    global_node = next((node for node in nodes if node.get("level") == "global"), {})
    return {
        "as_of": context_grid.get("as_of"),
        "dimensions": global_node.get("dimensions", {}),
        "status": context_grid.get("status", "unknown"),
    }


def _practical_implications(nodes: list[dict[str, Any]], result: dict[str, Any]) -> list[str]:
    implications = [
        "Keep the system in review-only mode until scenario outcomes are observed and approved.",
        "Treat historical analogs as retrieval candidates, not causal proof.",
    ]
    if nodes:
        top = nodes[0]
        implications.append(
            f"Highest-probability scenario is {top.get('label') or top.get('scenario_node_id')} at {float(top.get('probability', 0.0)):.1%}; monitor its invalidation conditions."
        )
    if not (result.get("pipeline_metric_snapshot") or {}):
        implications.append("Pipeline-derived metrics are absent; analytical conclusions rely on bounded evidence and qualitative context.")
    return implications
