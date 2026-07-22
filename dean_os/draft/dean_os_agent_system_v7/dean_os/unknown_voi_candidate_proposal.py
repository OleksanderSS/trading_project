from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso


class UnknownValueOfInformationCandidateProposalBuilder:
    """Select a small review set without inventing VoI component values or scores."""

    contract = "dean_unknown_voi_candidate_proposal_v1"

    def __init__(
        self,
        output_dir: str | Path = "reports/dean_os/unknown_voi_candidate_proposal_current",
    ) -> None:
        self.output_dir = Path(output_dir)

    def build(
        self,
        evidence_plan_path: str | Path,
        voi_review_path: str | Path,
        *,
        max_candidates: int = 3,
        save: bool = True,
    ) -> dict[str, Any]:
        if max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        plan_path = Path(evidence_plan_path)
        review_path = Path(voi_review_path)
        plan = _load(plan_path)
        review = _load(review_path)
        if plan.get("contract") != "dean_replay_outcome_evidence_plan_v1":
            raise ValueError("unsupported evidence plan contract")
        if review.get("contract") != "dean_unknown_voi_review_v1":
            raise ValueError("unsupported VoI review contract")
        bound = ((review.get("inputs") or {}).get("evidence_plan") or {}).get("sha256")
        if bound != _sha256(plan_path):
            raise ValueError("VoI review is not bound to evidence plan")

        aggregate = _aggregate_lanes(plan)
        unscored = {
            item["gap_id"]
            for item in review.get("gap_reviews") or []
            if (item.get("assessment") or {}).get("triage_score") is None
        }
        eligible = [item for item in aggregate.values() if item["gap_id"] in unscored]
        ordered = sorted(eligible, key=_selection_key)
        selected = [_proposal(item, rank=index + 1) for index, item in enumerate(ordered[:max_candidates])]

        created_at = utc_now_iso()
        run_id = "unknown_voi_candidate_proposal_" + created_at.replace(":", "").replace("+00:00", "Z")
        payload: dict[str, Any] = {
            "run_id": run_id,
            "created_at": created_at,
            "mode": "unknown_voi_candidate_proposal",
            "contract": self.contract,
            "inputs": {
                "evidence_plan": {"path": str(plan_path), "sha256": _sha256(plan_path)},
                "voi_review": {"path": str(review_path), "sha256": _sha256(review_path)},
            },
            "summary": {
                "eligible_unscored_gap_count": len(eligible),
                "selected_candidate_count": len(selected),
                "max_candidates": max_candidates,
                "validated_assessment_count": 0,
                "triage_score_count": 0,
                "collector_execution_allowed": False,
                "can_trade": False,
            },
            "selection_policy": {
                "first": "prefer gaps linked to more distinct hypotheses",
                "second": "prefer missing/context-only gaps over already partial evidence",
                "third": "prefer existing metric routes, then dedicated collector gaps, then source refresh, then outcome waiting",
                "fourth": "use repeated 30/90/180 references only as reach context, never as probability",
                "not_a_voi_score": True,
                "human_validation_required": True,
            },
            "candidates": selected,
            "safety": {
                "review_only": True,
                "automatic_voi_assessment_performed": False,
                "numeric_voi_values_inferred": False,
                "collector_task_created": False,
                "collector_execution_performed": False,
                "learning_write_performed": False,
                "can_trade": False,
            },
        }
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(self.output_dir).write(
                payload=payload, markdown=_markdown(payload), run_id=run_id
            )
        return payload


def _aggregate_lanes(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    aggregate: dict[str, dict[str, Any]] = {}
    hypotheses: dict[str, set[str]] = defaultdict(set)
    horizons: dict[str, set[int]] = defaultdict(set)
    refs: dict[str, int] = defaultdict(int)
    for task in plan.get("task_plans") or []:
        for lane in task.get("evidence_lanes") or []:
            gap_id = str(lane.get("gap_id") or "")
            if not gap_id:
                continue
            aggregate.setdefault(gap_id, dict(lane))
            hypotheses[gap_id].add(str(task.get("hypothesis_id") or ""))
            if task.get("horizon_days") is not None:
                horizons[gap_id].add(int(task["horizon_days"]))
            refs[gap_id] += 1
    for gap_id, item in aggregate.items():
        item["distinct_hypothesis_ids"] = sorted(value for value in hypotheses[gap_id] if value)
        item["horizons"] = sorted(horizons[gap_id])
        item["lane_reference_count"] = refs[gap_id]
    return aggregate


def _selection_key(item: dict[str, Any]) -> tuple[Any, ...]:
    status_order = {"missing": 0, "context_only_not_resolved": 1, "partial_supported": 2}
    route_order = {
        "route_available_metric_gap_open": 0,
        "structured_adapter_ready_source_feed_missing": 1,
        "dedicated_collector_missing": 1,
        "intake_path_available_source_refresh_required": 2,
        "route_available_outcome_not_matured": 3,
    }
    return (
        -len(item["distinct_hypothesis_ids"]),
        status_order.get(item.get("resolution_status"), 3),
        route_order.get((item.get("collection_route") or {}).get("status"), 4),
        -item["lane_reference_count"],
        item["gap_id"],
    )


def _proposal(item: dict[str, Any], *, rank: int) -> dict[str, Any]:
    return {
        "proposal_rank": rank,
        "gap_id": item["gap_id"],
        "description": item.get("description"),
        "resolution_status": item.get("resolution_status"),
        "expected_source_type": item.get("expected_source_type"),
        "collection_route": item.get("collection_route"),
        "distinct_hypothesis_ids": item["distinct_hypothesis_ids"],
        "horizons": item["horizons"],
        "lane_reference_count": item["lane_reference_count"],
        "selection_basis": [
            f"linked_hypothesis_count:{len(item['distinct_hypothesis_ids'])}",
            f"resolution_status:{item.get('resolution_status')}",
            f"collection_route_status:{(item.get('collection_route') or {}).get('status')}",
            f"lane_reference_count:{item['lane_reference_count']}",
        ],
        "assessment_status": "draft_review_candidate_not_assessed",
        "suggested_numeric_values": None,
        "triage_score": None,
        "review_required": True,
    }


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _markdown(payload: dict[str, Any]) -> str:
    lines = ["# Unknown VoI Candidate Proposal", ""]
    for item in payload["candidates"]:
        lines.append(f"- {item['proposal_rank']}. `{item['gap_id']}` — {item['description']}")
    lines.extend(["", "No VoI values or scores were inferred. Human validation is required."])
    return "\n".join(lines) + "\n"


__all__ = ["UnknownValueOfInformationCandidateProposalBuilder"]
