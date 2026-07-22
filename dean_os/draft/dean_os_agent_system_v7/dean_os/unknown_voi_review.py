from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.draft.dean_os_agent_system_v7.dean_os.unknown_graph import ValueOfInformationAssessment


class UnknownValueOfInformationReviewBuilder:
    contract = "dean_unknown_voi_review_v1"

    def __init__(
        self,
        output_dir: str | Path = "reports/dean_os/unknown_voi_review_current",
    ) -> None:
        self.output_dir = Path(output_dir)

    def build(
        self,
        evidence_plan_path: str | Path,
        *,
        assessments_path: str | Path | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        plan_path = Path(evidence_plan_path)
        plan = _load(plan_path)
        if plan.get("contract") != "dean_replay_outcome_evidence_plan_v1":
            raise ValueError("unsupported evidence plan contract")
        gaps = _unique_gap_lanes(plan)
        supplied = _load_assessments(assessments_path)
        unknown_ids = sorted(set(supplied) - set(gaps))
        if unknown_ids:
            raise ValueError(f"assessment references unknown gap ids: {unknown_ids}")

        reviews = []
        for gap_id, lane in sorted(gaps.items()):
            raw = supplied.get(gap_id)
            assessment = (
                ValueOfInformationAssessment.model_validate(raw).calculate()
                if raw is not None
                else ValueOfInformationAssessment()
            )
            reviews.append({
                "gap_id": gap_id,
                "description": lane.get("description"),
                "resolution_status": lane.get("resolution_status"),
                "expected_source_type": lane.get("expected_source_type"),
                "collection_route": lane.get("collection_route"),
                "linked_hypothesis_ids": (
                    lane.get("value_of_information") or {}
                ).get("linked_hypothesis_ids", []),
                "assessment": assessment.model_dump(mode="json"),
                "collector_execution_allowed": False,
                "manual_review_required": True,
            })

        ranked = sorted(
            [item for item in reviews if item["assessment"]["triage_score"] is not None],
            key=lambda item: (-item["assessment"]["triage_score"], item["gap_id"]),
        )
        created_at = utc_now_iso()
        run_id = "unknown_voi_review_" + created_at.replace(":", "").replace("+00:00", "Z")
        payload: dict[str, Any] = {
            "run_id": run_id,
            "created_at": created_at,
            "mode": "unknown_value_of_information_review",
            "contract": self.contract,
            "inputs": {
                "evidence_plan": {"path": str(plan_path), "sha256": _sha256(plan_path)},
                "assessments": (
                    {"path": str(assessments_path), "sha256": _sha256(Path(assessments_path))}
                    if assessments_path else None
                ),
            },
            "summary": {
                "unique_gap_count": len(reviews),
                "validated_scored_count": len(ranked),
                "unscored_count": len(reviews) - len(ranked),
                "collector_order_available": bool(ranked),
                "collector_execution_allowed": False,
                "can_trade": False,
            },
            "gap_reviews": reviews,
            "validated_collector_ranking": [
                {"rank": index + 1, "gap_id": item["gap_id"], "triage_score": item["assessment"]["triage_score"]}
                for index, item in enumerate(ranked)
            ],
            "review_questions": [
                "Could this evidence change scenario probabilities or scenario selection?",
                "Could it materially change estimate confidence?",
                "Could it block a wrong conclusion or unsafe readiness state?",
                "Is it decision-relevant for a linked hypothesis?",
                "How feasible and costly is point-in-time collection?",
                "What evidence supports each assessment component?",
            ],
            "safety": {
                "review_only": True,
                "automatic_assessment_performed": False,
                "collector_task_created": False,
                "collector_execution_performed": False,
                "gap_closed": False,
                "learning_write_performed": False,
                "can_trade": False,
            },
        }
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(self.output_dir).write(
                payload=payload, markdown=_markdown(payload), run_id=run_id
            )
        return payload


def _unique_gap_lanes(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    gaps: dict[str, dict[str, Any]] = {}
    for task in plan.get("task_plans") or []:
        for lane in task.get("evidence_lanes") or []:
            gap_id = str(lane.get("gap_id") or "")
            if gap_id:
                gaps.setdefault(gap_id, lane)
    return gaps


def _load_assessments(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = _load(Path(path))
    rows = payload.get("assessments")
    if not isinstance(rows, list):
        raise ValueError("assessment input requires assessments[]")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get("gap_id"):
            raise ValueError("every assessment requires gap_id")
        gap_id = str(row["gap_id"])
        if gap_id in result:
            raise ValueError(f"duplicate assessment gap_id: {gap_id}")
        result[gap_id] = {key: value for key, value in row.items() if key != "gap_id"}
    return result


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    return (
        "# Unknown Value of Information Review\n\n"
        f"- Unique gaps: `{summary['unique_gap_count']}`\n"
        f"- Validated/scored: `{summary['validated_scored_count']}`\n"
        f"- Unscored: `{summary['unscored_count']}`\n"
        f"- Collector execution allowed: `false`\n"
        f"- Can trade: `false`\n"
    )


__all__ = ["UnknownValueOfInformationReviewBuilder"]
