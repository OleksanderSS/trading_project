from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.draft.dean_os_agent_system_v7.dean_os.paper_lifecycle_contract import (
    PAPER_LIFECYCLE_SCHEMA_VERSION,
    file_sha256,
    fingerprint_matches,
    load_json_object,
    object_fingerprint,
    valid_sha256,
)
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

PostPaperSimulationDecision = Literal[
    "ready_for_human_review",
    "rerun_paper_simulation",
    "reject",
    "needs_more_data",
]

PostPaperSimulationVerdict = Literal[
    "clear",
    "caution",
    "blocked",
    "needs_more_data",
]


class PostPaperSimulationReviewArtifact(BaseModel):
    """Supervised review artifact after PaperSimulationResult.

    This layer decides what a human should inspect next. It does not authorize
    live execution, broker access, model promotion, production config writes, or
    learning-memory writes.
    """

    review_id: str = Field(default_factory=lambda: f"post_paper_sim_review_{uuid4().hex}")
    created_at: str = Field(default_factory=utc_now_iso)
    decision: PostPaperSimulationDecision
    verdict: PostPaperSimulationVerdict
    confidence: float = Field(ge=0.0, le=1.0)
    data_quality_score: float = Field(ge=0.0, le=1.0)
    source_result_id: str | None = None
    source_result_path: str | None = None
    source_result_sha256: str | None = None
    source_result_fingerprint: str | None = None
    source_result_status: str | None = None
    source_result_recommendation: str | None = None
    reasons: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    required_followups: list[str] = Field(default_factory=list)
    guardrail_summary: dict[str, bool] = Field(default_factory=dict)
    metric_summary: dict[str, Any] = Field(default_factory=dict)
    lineage_verified: bool = False

    review_required: bool = True
    live_execution_allowed: bool = False
    broker_access_allowed: bool = False
    production_config_write_allowed: bool = False
    learning_memory_write_allowed: bool = False
    model_promotion_allowed: bool = False
    live_candidate_allowed: bool = False


class PostPaperSimulationReviewBuilder:
    """Builds supervised post-paper-simulation review artifacts."""

    def __init__(
        self,
        paper_simulation_result_path: str | Path,
        output_dir: str | Path = "reports/dean_os/post_paper_simulation_review",
    ):
        self.paper_simulation_result_path = Path(paper_simulation_result_path)
        self.output_dir = Path(output_dir)

    def build(self, save: bool = True) -> dict[str, Any]:
        result_payload = self._load_result()
        review = build_post_paper_simulation_review(
            result_payload=result_payload,
            source_result_path=str(self.paper_simulation_result_path),
        )
        review_payload = review.model_dump(mode="json")
        payload = {
            "run_id": review.review_id,
            "mode": "post_paper_simulation_review",
            "schema_version": PAPER_LIFECYCLE_SCHEMA_VERSION,
            "created_at": utc_now_iso(),
            "post_paper_simulation_review": review_payload,
            "review_fingerprint": object_fingerprint(review_payload),
            "source_result_summary": _result_summary(result_payload),
            "safety": {
                "review_only": True,
                "live_execution_allowed": False,
                "broker_access_allowed": False,
                "production_config_write_allowed": False,
                "learning_memory_write_allowed": False,
                "model_promotion_allowed": False,
                "live_candidate_allowed": False,
                "approval_performed": False,
                "lineage_verified": review.lineage_verified,
            },
        }

        if save:
            markdown = render_post_paper_simulation_review_markdown(payload)
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=markdown,
                run_id=review.review_id,
            )
            payload["saved_paths"] = saved_paths

        return json_ready(payload)

    def _load_result(self) -> dict[str, Any]:
        if not self.paper_simulation_result_path.exists():
            return {
                "mode": "missing_paper_simulation_result",
                "paper_simulation_result": None,
                "errors": [f"Paper simulation result not found: {self.paper_simulation_result_path}"],
            }

        try:
            payload = json.loads(self.paper_simulation_result_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {
                "mode": "invalid_paper_simulation_result",
                "paper_simulation_result": None,
                "errors": [repr(exc)],
            }

        if not isinstance(payload, dict):
            return {
                "mode": "invalid_paper_simulation_result",
                "paper_simulation_result": None,
                "errors": ["Paper simulation result JSON is not an object"],
            }
        return payload


def build_post_paper_simulation_review(
    *,
    result_payload: dict[str, Any],
    source_result_path: str | None = None,
) -> PostPaperSimulationReviewArtifact:
    result = result_payload.get("paper_simulation_result")
    if not isinstance(result, dict):
        return PostPaperSimulationReviewArtifact(
            decision="needs_more_data",
            verdict="needs_more_data",
            confidence=0.72,
            data_quality_score=0.25,
            source_result_path=source_result_path,
            reasons=["No valid PaperSimulationResult artifact was supplied."],
            risks=_base_risks(),
            next_actions=["Record an isolated paper simulation result before post-paper review."],
            required_followups=["Provide reports/dean_os/paper_simulation_results/latest.json."],
        )

    result_sha256 = file_sha256(source_result_path)
    result_fingerprint = result_payload.get("result_fingerprint")
    lineage_issues = _result_lineage_issues(
        result_payload=result_payload,
        source_result_path=source_result_path,
    )
    if lineage_issues:
        return PostPaperSimulationReviewArtifact(
            decision="reject",
            verdict="blocked",
            confidence=0.95,
            data_quality_score=0.2,
            source_result_id=result.get("result_id"),
            source_result_path=source_result_path,
            source_result_sha256=result_sha256,
            source_result_fingerprint=result_fingerprint,
            source_result_status=result.get("status"),
            source_result_recommendation=result.get(
                "recommendation"
            ),
            reasons=[
                "Paper simulation result lineage is invalid.",
                *lineage_issues,
            ],
            risks=_base_risks(),
            next_actions=[
                "Reject this result and rebuild the receipt/plan/external "
                "result chain without changing bound artifacts."
            ],
            required_followups=[
                "Provide a hash-bound result from one validated isolated "
                "executor output."
            ],
            guardrail_summary=_guardrail_summary(result),
            metric_summary=_metric_summary(result),
            lineage_verified=False,
        )

    unsafe = _unsafe_result_flags(result, result_payload)
    if unsafe:
        return PostPaperSimulationReviewArtifact(
            decision="reject",
            verdict="blocked",
            confidence=0.93,
            data_quality_score=0.35,
            source_result_id=result.get("result_id"),
            source_result_path=source_result_path,
            source_result_sha256=result_sha256,
            source_result_fingerprint=result_fingerprint,
            source_result_status=result.get("status"),
            source_result_recommendation=result.get("recommendation"),
            reasons=["Paper simulation result contains unsafe side-effect flags.", *unsafe],
            risks=_base_risks(),
            next_actions=["Reject this paper simulation result and inspect the isolated executor/environment."],
            required_followups=["Produce a clean PaperSimulationResult with all live/broker/prod/learning/promotion flags false."],
            guardrail_summary=_guardrail_summary(result),
            metric_summary=_metric_summary(result),
        )

    status = str(result.get("status") or "")
    recommendation = str(result.get("recommendation") or "")
    guardrails = _guardrail_summary(result)
    guardrails_ok = bool(guardrails) and all(guardrails.values())
    metrics = _metric_summary(result)
    warnings = list(result.get("warnings") or [])
    errors = list(result.get("errors") or [])

    reasons = [f"Paper simulation result status={status}, recommendation={recommendation}."]
    if errors:
        reasons.append(f"Paper simulation result reported {len(errors)} error(s).")
    if warnings:
        reasons.append(f"Paper simulation result reported {len(warnings)} warning(s).")
    if guardrails and not guardrails_ok:
        failed = [key for key, value in guardrails.items() if value is False]
        reasons.append(f"Guardrail checks failed: {', '.join(failed)}.")
    if metrics:
        reasons.append(f"Metric summary available: {', '.join(sorted(metrics.keys()))}.")

    if status in {
        "blocked_invalid_plan",
        "blocked_unsafe_plan",
        "blocked_missing_external_evidence",
        "blocked_external_evidence_mismatch",
    } or recommendation == "reject" or errors:
        decision = "reject"
        verdict = "blocked"
        confidence = 0.88
        data_quality = 0.45
        next_actions = [
            "Reject this paper simulation result.",
            "Inspect errors and source paper simulation plan before any rerun.",
        ]
        followups = ["Fix paper simulation plan/result blockers and create a new review receipt if needed."]
    elif status == "failed":
        decision = "reject"
        verdict = "blocked"
        confidence = 0.82
        data_quality = 0.5
        next_actions = [
            "Reject failed paper simulation.",
            "Inspect isolated paper executor logs and decide whether to create a new paper simulation plan.",
        ]
        followups = ["Attach paper simulation logs/errors before any rerun."]
    elif recommendation == "rerun_paper_simulation" or status == "completed_with_warnings" or warnings or (guardrails and not guardrails_ok):
        decision = "rerun_paper_simulation"
        verdict = "caution"
        confidence = 0.78
        data_quality = 0.62
        next_actions = [
            "Review warnings/guardrail gaps and rerun paper simulation after fixes.",
            "Do not progress to model promotion or live-candidate discussion from this result.",
        ]
        followups = ["Resolve warnings and failed guardrail checks."]
    elif recommendation == "ready_for_review" and status == "completed" and (not guardrails or guardrails_ok):
        decision = "ready_for_human_review"
        verdict = "clear"
        confidence = 0.84
        data_quality = 0.78
        next_actions = [
            "Human reviewer can inspect paper simulation result and decide whether it remains useful for research/process learning.",
            "No automatic model promotion, production config change, broker/live action, or learning write is authorized.",
        ]
        followups = [
            "If accepted, create a separate human review receipt for any next non-live process step.",
            "Keep live execution blocked unless a future dedicated live-readiness process is designed and reviewed.",
        ]
    else:
        decision = "needs_more_data"
        verdict = "needs_more_data"
        confidence = 0.7
        data_quality = 0.45
        next_actions = ["Provide clearer paper simulation result status, recommendation, metrics, and guardrail checks."]
        followups = ["Backfill missing paper simulation result details."]

    return PostPaperSimulationReviewArtifact(
        decision=decision,
        verdict=verdict,
        confidence=confidence,
        data_quality_score=data_quality,
        source_result_id=result.get("result_id"),
        source_result_path=source_result_path,
        source_result_sha256=result_sha256,
        source_result_fingerprint=result_fingerprint,
        source_result_status=status,
        source_result_recommendation=recommendation,
        reasons=reasons,
        risks=_base_risks(),
        next_actions=next_actions,
        required_followups=followups,
        guardrail_summary=guardrails,
        metric_summary=metrics,
        lineage_verified=True,
        review_required=True,
        live_execution_allowed=False,
        broker_access_allowed=False,
        production_config_write_allowed=False,
        learning_memory_write_allowed=False,
        model_promotion_allowed=False,
        live_candidate_allowed=False,
    )


def render_post_paper_simulation_review_markdown(payload: dict[str, Any]) -> str:
    review = payload.get("post_paper_simulation_review") or {}
    source = payload.get("source_result_summary") or {}
    lines = [
        "# DEAN-OS Post Paper Simulation Review",
        "",
        f"- Review ID: `{review.get('review_id')}`",
        f"- Decision: `{review.get('decision')}`",
        f"- Verdict: `{review.get('verdict')}`",
        f"- Confidence: `{review.get('confidence')}`",
        f"- Data quality: `{review.get('data_quality_score')}`",
        f"- Source result: `{review.get('source_result_id')}`",
        f"- Source result SHA256: `{review.get('source_result_sha256')}`",
        f"- Lineage verified: `{review.get('lineage_verified')}`",
        f"- Source status: `{review.get('source_result_status')}`",
        f"- Source recommendation: `{review.get('source_result_recommendation')}`",
        "",
        "## Source Result Summary",
        "",
    ]

    for key, value in source.items():
        lines.append(f"- {key}: `{value}`")

    lines.extend(["", "## Reasons", ""])
    for item in review.get("reasons") or []:
        lines.append(f"- {item}")

    lines.extend(["", "## Metrics", ""])
    for key, value in sorted((review.get("metric_summary") or {}).items()):
        lines.append(f"- {key}: `{value}`")
    if not review.get("metric_summary"):
        lines.append("- No metric summary supplied.")

    lines.extend(["", "## Guardrail Summary", ""])
    for key, value in sorted((review.get("guardrail_summary") or {}).items()):
        lines.append(f"- {key}: `{value}`")
    if not review.get("guardrail_summary"):
        lines.append("- No guardrail checks supplied.")

    lines.extend(["", "## Next Actions", ""])
    for item in review.get("next_actions") or []:
        lines.append(f"- {item}")

    lines.extend(["", "## Required Follow-ups", ""])
    for item in review.get("required_followups") or []:
        lines.append(f"- {item}")

    lines.extend(["", "## Risks", ""])
    for item in review.get("risks") or []:
        lines.append(f"- {item}")

    lines.extend(["", "## Safety", ""])
    safety = payload.get("safety") or {}
    for key in sorted(safety):
        lines.append(f"- {key}: `{safety[key]}`")

    lines.extend(
        [
            "",
            "## Operator Note",
            "",
            "This review does not authorize live execution, broker access, model promotion, production config writes, or learning-memory writes.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _unsafe_result_flags(result: dict[str, Any], payload: dict[str, Any]) -> list[str]:
    dangerous = {
        "live_execution_allowed",
        "broker_access_allowed",
        "production_config_write_allowed",
        "learning_memory_write_allowed",
        "model_promotion_allowed",
    }
    messages: list[str] = []
    safety = dict(payload.get("safety") or {})
    for flag in dangerous:
        if result.get(flag) is True:
            messages.append(f"paper_simulation_result.{flag}=true")
        if safety.get(flag) is True:
            messages.append(f"safety.{flag}=true")
    if result.get("paper_simulation_executed_by_this_layer") is True or safety.get("paper_simulation_executed_by_this_layer") is True:
        messages.append("paper_simulation_executed_by_this_layer=true")
    return messages


def _guardrail_summary(result: dict[str, Any]) -> dict[str, bool]:
    raw = result.get("guardrail_checks") or {}
    if not isinstance(raw, dict):
        return {}
    return {str(key): bool(value) for key, value in raw.items()}


def _metric_summary(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("metrics") or []
    summary: dict[str, Any] = {}
    if isinstance(metrics, list):
        for metric in metrics:
            if isinstance(metric, dict) and metric.get("name"):
                summary[str(metric["name"])] = metric.get("value")
    return summary


def _result_summary(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload.get("paper_simulation_result") if isinstance(payload.get("paper_simulation_result"), dict) else {}
    metrics = result.get("metrics") or []
    return {
        "mode": payload.get("mode"),
        "result_id": result.get("result_id"),
        "status": result.get("status"),
        "recommendation": result.get("recommendation"),
        "source_plan_id": result.get("source_plan_id"),
        "source_receipt_id": result.get("source_receipt_id"),
        "metric_count": len(metrics) if isinstance(metrics, list) else 0,
        "warning_count": len(result.get("warnings") or []),
        "error_count": len(result.get("errors") or []),
    }


def _base_risks() -> list[str]:
    return [
        "Paper simulation success does not authorize live execution.",
        "Paper simulation success does not authorize model promotion.",
        "Human review is required before any further process step.",
        "Broker/live credentials must remain unavailable to this workflow.",
    ]


def _result_lineage_issues(
    *,
    result_payload: dict[str, Any],
    source_result_path: str | None,
) -> list[str]:
    issues: list[str] = []
    if result_payload.get("mode") != "paper_simulation_result":
        issues.append("result_mode_invalid")
    if (
        result_payload.get("schema_version")
        != PAPER_LIFECYCLE_SCHEMA_VERSION
    ):
        issues.append("result_schema_version_invalid")
    if not fingerprint_matches(
        result_payload,
        object_key="paper_simulation_result",
        fingerprint_key="result_fingerprint",
    ):
        issues.append("result_fingerprint_invalid")
    if not valid_sha256(file_sha256(source_result_path)):
        issues.append("result_file_sha256_unavailable")
    result = result_payload.get("paper_simulation_result") or {}
    if result.get("lineage_verified") is not True:
        issues.append("result_lineage_not_verified")
    if not valid_sha256(result.get("source_plan_sha256")):
        issues.append("result_source_plan_sha256_missing")
    elif (
        file_sha256(result.get("source_plan_path"))
        != result.get("source_plan_sha256")
    ):
        issues.append("result_source_plan_sha256_mismatch")
    if not valid_sha256(result.get("external_result_sha256")):
        issues.append("result_external_result_sha256_missing")
    elif (
        file_sha256(result.get("external_result_path"))
        != result.get("external_result_sha256")
    ):
        issues.append("result_external_result_sha256_mismatch")
    if not valid_sha256(result.get("external_result_fingerprint")):
        issues.append("result_external_result_fingerprint_missing")
    external_payload = load_json_object(
        result.get("external_result_path")
    )
    if external_payload is None:
        issues.append("result_external_result_unavailable")
    elif (
        external_payload.get("output_fingerprint")
        != result.get("external_result_fingerprint")
    ):
        issues.append("result_external_result_fingerprint_mismatch")
    return sorted(set(issues))
