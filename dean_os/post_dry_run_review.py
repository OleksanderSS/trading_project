from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

PostDryRunDecision = Literal[
    "ready_for_human_review",
    "rerun_dry_run",
    "reject",
    "needs_more_data",
]

PostDryRunVerdict = Literal[
    "clear",
    "caution",
    "blocked",
    "needs_more_data",
]


class PostDryRunReviewArtifact(BaseModel):
    """Supervised review artifact after a DryRunResult.

    This layer decides what a human should do next. It does not approve paper
    execution, live execution, model promotion, config writes, or learning writes.
    """

    review_id: str = Field(default_factory=lambda: f"post_dry_run_review_{uuid4().hex}")
    created_at: str = Field(default_factory=utc_now_iso)
    decision: PostDryRunDecision
    verdict: PostDryRunVerdict
    confidence: float = Field(ge=0.0, le=1.0)
    data_quality_score: float = Field(ge=0.0, le=1.0)
    source_result_id: str | None = None
    source_result_path: str | None = None
    source_result_status: str | None = None
    source_result_recommendation: str | None = None
    reasons: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    required_followups: list[str] = Field(default_factory=list)
    guardrail_summary: dict[str, bool] = Field(default_factory=dict)

    review_required: bool = True
    paper_simulation_allowed: bool = False
    live_execution_allowed: bool = False
    broker_access_allowed: bool = False
    production_config_write_allowed: bool = False
    learning_memory_write_allowed: bool = False
    model_promotion_allowed: bool = False


class PostDryRunReviewBuilder:
    """Builds a post dry-run review from a DryRunResult artifact."""

    def __init__(
        self,
        dry_run_result_path: str | Path,
        output_dir: str | Path = "reports/dean_os/post_dry_run_review",
    ):
        self.dry_run_result_path = Path(dry_run_result_path)
        self.output_dir = Path(output_dir)

    def build(self, save: bool = True) -> dict[str, Any]:
        result_payload = self._load_result()
        review = build_post_dry_run_review(
            result_payload=result_payload,
            source_result_path=str(self.dry_run_result_path),
        )
        payload = {
            "run_id": review.review_id,
            "mode": "post_dry_run_review",
            "created_at": utc_now_iso(),
            "post_dry_run_review": review.model_dump(mode="json"),
            "source_result_summary": _result_summary(result_payload),
            "safety": {
                "review_only": True,
                "paper_simulation_allowed": False,
                "live_execution_allowed": False,
                "broker_access_allowed": False,
                "production_config_write_allowed": False,
                "learning_memory_write_allowed": False,
                "model_promotion_allowed": False,
                "approval_performed": False,
            },
        }

        if save:
            markdown = render_post_dry_run_review_markdown(payload)
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=markdown,
                run_id=review.review_id,
            )
            payload["saved_paths"] = saved_paths

        return json_ready(payload)

    def _load_result(self) -> dict[str, Any]:
        if not self.dry_run_result_path.exists():
            return {
                "mode": "missing_dry_run_result",
                "dry_run_result": None,
                "errors": [f"Dry-run result not found: {self.dry_run_result_path}"],
            }

        try:
            payload = json.loads(self.dry_run_result_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {
                "mode": "invalid_dry_run_result",
                "dry_run_result": None,
                "errors": [repr(exc)],
            }

        if not isinstance(payload, dict):
            return {
                "mode": "invalid_dry_run_result",
                "dry_run_result": None,
                "errors": ["Dry-run result JSON is not an object"],
            }
        return payload


def build_post_dry_run_review(
    *,
    result_payload: dict[str, Any],
    source_result_path: str | None = None,
) -> PostDryRunReviewArtifact:
    result = result_payload.get("dry_run_result")
    if not isinstance(result, dict):
        return PostDryRunReviewArtifact(
            decision="needs_more_data",
            verdict="needs_more_data",
            confidence=0.72,
            data_quality_score=0.25,
            source_result_path=source_result_path,
            reasons=["No valid DryRunResult artifact was supplied."],
            risks=_base_risks(),
            next_actions=["Run or record an isolated dry-run result before post-dry-run review."],
            required_followups=["Provide reports/dean_os/dry_run_results/latest.json."],
        )

    unsafe = _unsafe_result_flags(result, result_payload)
    if unsafe:
        return PostDryRunReviewArtifact(
            decision="reject",
            verdict="blocked",
            confidence=0.92,
            data_quality_score=0.35,
            source_result_id=result.get("result_id"),
            source_result_path=source_result_path,
            source_result_status=result.get("status"),
            source_result_recommendation=result.get("recommendation"),
            reasons=["Dry-run result contains unsafe side-effect flags.", *unsafe],
            risks=_base_risks(),
            next_actions=["Reject this dry-run result and inspect the isolated executor/environment."],
            required_followups=["Produce a clean DryRunResult with all live/prod/learning/promotion flags false."],
            guardrail_summary=_guardrail_summary(result),
        )

    status = str(result.get("status") or "")
    recommendation = str(result.get("recommendation") or "")
    guardrails = _guardrail_summary(result)
    guardrails_ok = bool(guardrails) and all(guardrails.values())

    reasons: list[str] = [
        f"Dry-run result status={status}, recommendation={recommendation}.",
    ]
    warnings = list(result.get("warnings") or [])
    errors = list(result.get("errors") or [])

    if errors:
        reasons.append(f"Dry-run result reported {len(errors)} error(s).")
    if warnings:
        reasons.append(f"Dry-run result reported {len(warnings)} warning(s).")
    if guardrails and not guardrails_ok:
        failed = [key for key, value in guardrails.items() if value is False]
        reasons.append(f"Guardrail checks failed: {', '.join(failed)}.")

    if status in {"blocked_invalid_plan", "blocked_unsafe_plan"} or recommendation == "reject" or errors:
        decision = "reject"
        verdict = "blocked"
        confidence = 0.88
        data_quality = 0.45
        next_actions = [
            "Reject this dry-run result.",
            "Inspect errors and source dry-run plan before any rerun.",
        ]
        followups = ["Fix dry-run plan/result blockers and create a new review receipt if needed."]
    elif status == "failed":
        decision = "reject"
        verdict = "blocked"
        confidence = 0.82
        data_quality = 0.5
        next_actions = [
            "Reject failed dry-run.",
            "Inspect isolated executor logs and decide whether to create a new dry-run plan.",
        ]
        followups = ["Attach executor logs/errors before any rerun."]
    elif recommendation == "rerun_dry_run" or status == "completed_with_warnings" or warnings or (guardrails and not guardrails_ok):
        decision = "rerun_dry_run"
        verdict = "caution"
        confidence = 0.78
        data_quality = 0.62
        next_actions = [
            "Review warnings/guardrail gaps and rerun dry-run after fixes.",
            "Do not progress to paper-only simulation from this result.",
        ]
        followups = ["Resolve warnings and failed guardrail checks."]
    elif recommendation == "ready_for_review" and status == "completed" and (not guardrails or guardrails_ok):
        decision = "ready_for_human_review"
        verdict = "clear"
        confidence = 0.84
        data_quality = 0.78
        next_actions = [
            "Human reviewer can inspect dry-run result and decide whether to create a paper-only simulation receipt.",
            "No automatic paper/live action is authorized by this review.",
        ]
        followups = ["If accepted, create a separate ReviewDecisionReceipt for approve_paper_only_simulation."]
    else:
        decision = "needs_more_data"
        verdict = "needs_more_data"
        confidence = 0.7
        data_quality = 0.45
        next_actions = ["Provide clearer dry-run result status, recommendation, metrics, and guardrail checks."]
        followups = ["Backfill missing dry-run result details."]

    return PostDryRunReviewArtifact(
        decision=decision,
        verdict=verdict,
        confidence=confidence,
        data_quality_score=data_quality,
        source_result_id=result.get("result_id"),
        source_result_path=source_result_path,
        source_result_status=status,
        source_result_recommendation=recommendation,
        reasons=reasons,
        risks=_base_risks(),
        next_actions=next_actions,
        required_followups=followups,
        guardrail_summary=guardrails,
        review_required=True,
        paper_simulation_allowed=False,
        live_execution_allowed=False,
        broker_access_allowed=False,
        production_config_write_allowed=False,
        learning_memory_write_allowed=False,
        model_promotion_allowed=False,
    )


def render_post_dry_run_review_markdown(payload: dict[str, Any]) -> str:
    review = payload.get("post_dry_run_review") or {}
    source = payload.get("source_result_summary") or {}
    lines = [
        "# DEAN-OS Post Dry-Run Review",
        "",
        f"- Review ID: `{review.get('review_id')}`",
        f"- Decision: `{review.get('decision')}`",
        f"- Verdict: `{review.get('verdict')}`",
        f"- Confidence: `{review.get('confidence')}`",
        f"- Data quality: `{review.get('data_quality_score')}`",
        f"- Source result: `{review.get('source_result_id')}`",
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
            "This review does not authorize paper simulation, live execution, broker access, model promotion, production config writes, or learning-memory writes.",
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
            messages.append(f"dry_run_result.{flag}=true")
        if safety.get(flag) is True:
            messages.append(f"safety.{flag}=true")
    # This layer should only receive results from an external executor; it still
    # rejects if this layer claims it executed the dry-run itself.
    if result.get("dry_run_executed_by_this_layer") is True or safety.get("dry_run_executed_by_this_layer") is True:
        messages.append("dry_run_executed_by_this_layer=true")
    return messages


def _guardrail_summary(result: dict[str, Any]) -> dict[str, bool]:
    raw = result.get("guardrail_checks") or {}
    if not isinstance(raw, dict):
        return {}
    return {str(key): bool(value) for key, value in raw.items()}


def _result_summary(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload.get("dry_run_result") if isinstance(payload.get("dry_run_result"), dict) else {}
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
        "Dry-run success does not authorize paper simulation automatically.",
        "Human review is required before paper-only simulation.",
        "Live execution remains forbidden.",
        "Model promotion and production config writes remain forbidden.",
    ]
