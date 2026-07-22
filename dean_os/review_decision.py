from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.paper_lifecycle_contract import (
    PAPER_LIFECYCLE_SCHEMA_VERSION,
    file_sha256,
    object_fingerprint,
    parse_timestamp,
    valid_sha256,
)
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

ReviewDecisionType = Literal[
    "mark_reviewed",
    "needs_more_data",
    "reject",
    "approve_dry_run",
    "approve_paper_only_simulation",
]

ReviewDecisionStatus = Literal[
    "reviewed",
    "needs_more_data",
    "rejected",
    "dry_run_approved",
    "paper_only_approved",
]


class ReviewDecisionReceipt(BaseModel):
    """Human-review receipt for DEAN-OS review artifacts.

    This is an explicit human decision record. It does not authorize live trading,
    broker access, production config writes, model promotion, or automatic learning writes.
    """

    receipt_id: str = Field(default_factory=lambda: f"review_receipt_{uuid4().hex}")
    created_at: str = Field(default_factory=utc_now_iso)
    reviewer: str
    decision: ReviewDecisionType
    status: ReviewDecisionStatus
    rationale: str
    source_artifact_path: str
    source_artifact_mode: str | None = None
    source_artifact_run_id: str | None = None
    source_artifact_sha256: str | None = None
    source_decision: str | None = None
    source_verdict: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    required_followups: list[str] = Field(default_factory=list)
    scope: Literal["review_only", "dry_run", "paper_only"] = "review_only"
    expires_at: str | None = None

    review_required_after_receipt: bool = True
    live_execution_allowed: bool = False
    broker_access_allowed: bool = False
    production_config_write_allowed: bool = False
    learning_memory_write_allowed: bool = False
    model_promotion_allowed: bool = False

    @model_validator(mode="after")
    def validate_safety(self) -> ReviewDecisionReceipt:
        self.reviewer = self.reviewer.strip()
        self.rationale = self.rationale.strip()
        if not self.reviewer:
            raise ValueError("ReviewDecisionReceipt.reviewer cannot be empty")
        if not self.rationale:
            raise ValueError("ReviewDecisionReceipt.rationale cannot be empty")
        if self.decision == "approve_dry_run":
            self.status = "dry_run_approved"
            self.scope = "dry_run"
        elif self.decision == "approve_paper_only_simulation":
            self.status = "paper_only_approved"
            self.scope = "paper_only"
        elif self.decision == "needs_more_data":
            self.status = "needs_more_data"
            self.scope = "review_only"
        elif self.decision == "reject":
            self.status = "rejected"
            self.scope = "review_only"
        elif self.decision == "mark_reviewed":
            self.status = "reviewed"
            self.scope = "review_only"

        # Never allow live/prod side effects in this receipt layer.
        if self.live_execution_allowed:
            raise ValueError("ReviewDecisionReceipt cannot allow live execution")
        if self.broker_access_allowed:
            raise ValueError("ReviewDecisionReceipt cannot allow broker access")
        if self.production_config_write_allowed:
            raise ValueError("ReviewDecisionReceipt cannot allow production config writes")
        if self.learning_memory_write_allowed:
            raise ValueError("ReviewDecisionReceipt cannot allow learning memory writes")
        if self.model_promotion_allowed:
            raise ValueError("ReviewDecisionReceipt cannot allow model promotion")

        if self.decision in {"needs_more_data", "reject"} and not self.required_followups:
            self.required_followups.append("Reviewer requested no automatic follow-up beyond recording this decision.")
        if self.decision in {"approve_dry_run", "approve_paper_only_simulation"}:
            if parse_timestamp(self.expires_at) is None:
                raise ValueError(
                    "Approval receipts require a timezone-aware expires_at"
                )
            if not self.constraints:
                self.constraints.extend(
                    [
                        "human_review_required_before_any_promotion",
                        "no_live_execution",
                        "no_broker_access",
                        "no_production_config_write",
                        "source_artifact_hash_bound",
                    ]
                )
        if self.decision == "approve_paper_only_simulation":
            if self.source_artifact_mode != "post_dry_run_review":
                raise ValueError(
                    "Paper-only approval requires a post_dry_run_review source"
                )
            if not valid_sha256(self.source_artifact_sha256):
                raise ValueError(
                    "Paper-only approval requires source_artifact_sha256"
                )
            if self.source_decision != "ready_for_human_review":
                raise ValueError(
                    "Paper-only approval requires ready_for_human_review source"
                )
            if self.source_verdict not in {"clear", "caution"}:
                raise ValueError(
                    "Paper-only approval requires clear/caution source verdict"
                )
        return self


class ReviewDecisionRecorder:
    """Creates and persists human review receipts."""

    def __init__(
        self,
        output_dir: str | Path = "reports/dean_os/review_decisions",
    ):
        self.output_dir = Path(output_dir)

    def create_receipt(
        self,
        *,
        reviewer: str,
        decision: ReviewDecisionType,
        rationale: str,
        source_artifact_path: str | Path,
        evidence_refs: list[str] | None = None,
        constraints: list[str] | None = None,
        required_followups: list[str] | None = None,
        expires_at: str | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        source = _load_source_artifact(source_artifact_path)
        receipt = ReviewDecisionReceipt(
            reviewer=reviewer,
            decision=decision,
            status=_status_for(decision),
            rationale=rationale,
            source_artifact_path=str(source_artifact_path),
            source_artifact_mode=source.get("mode"),
            source_artifact_run_id=source.get("run_id"),
            source_artifact_sha256=file_sha256(source_artifact_path),
            source_decision=_source_decision(source),
            source_verdict=_source_verdict(source),
            evidence_refs=evidence_refs or [],
            constraints=constraints or [],
            required_followups=required_followups or [],
            expires_at=expires_at,
        )
        receipt_payload = receipt.model_dump(mode="json")
        payload = {
            "run_id": receipt.receipt_id,
            "mode": "review_decision_receipt",
            "schema_version": PAPER_LIFECYCLE_SCHEMA_VERSION,
            "receipt": receipt_payload,
            "receipt_fingerprint": object_fingerprint(receipt_payload),
            "source_summary": _source_summary(source),
            "safety": {
                "review_only": True,
                "human_decision_recorded": True,
                "approval_performed": decision in {"approve_dry_run", "approve_paper_only_simulation"},
                "dry_run_approved": decision == "approve_dry_run",
                "paper_only_simulation_approved": decision == "approve_paper_only_simulation",
                "live_execution_allowed": False,
                "broker_access_allowed": False,
                "production_config_write_allowed": False,
                "learning_memory_write_allowed": False,
                "model_promotion_allowed": False,
            },
        }

        if save:
            markdown = render_review_decision_markdown(payload)
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=markdown,
                run_id=receipt.receipt_id,
            )
            payload["saved_paths"] = saved_paths

        return json_ready(payload)


def render_review_decision_markdown(payload: dict[str, Any]) -> str:
    receipt = payload.get("receipt") or {}
    source = payload.get("source_summary") or {}
    safety = payload.get("safety") or {}
    lines = [
        "# DEAN-OS Review Decision Receipt",
        "",
        f"- Receipt ID: `{receipt.get('receipt_id')}`",
        f"- Created at: `{receipt.get('created_at')}`",
        f"- Reviewer: `{receipt.get('reviewer')}`",
        f"- Decision: `{receipt.get('decision')}`",
        f"- Status: `{receipt.get('status')}`",
        f"- Scope: `{receipt.get('scope')}`",
        f"- Source artifact: `{receipt.get('source_artifact_path')}`",
        f"- Source artifact mode: `{receipt.get('source_artifact_mode')}`",
        f"- Source run ID: `{receipt.get('source_artifact_run_id')}`",
        f"- Source SHA256: `{receipt.get('source_artifact_sha256')}`",
        f"- Source decision: `{receipt.get('source_decision')}`",
        f"- Source verdict: `{receipt.get('source_verdict')}`",
        "",
        "## Rationale",
        "",
        str(receipt.get("rationale") or ""),
        "",
        "## Source Summary",
        "",
    ]

    for key, value in source.items():
        lines.append(f"- {key}: `{value}`")

    lines.extend(["", "## Evidence References", ""])
    for ref in receipt.get("evidence_refs") or []:
        lines.append(f"- {ref}")
    if not receipt.get("evidence_refs"):
        lines.append("- None supplied.")

    lines.extend(["", "## Constraints", ""])
    for item in receipt.get("constraints") or []:
        lines.append(f"- {item}")
    if not receipt.get("constraints"):
        lines.append("- No additional constraints supplied.")

    lines.extend(["", "## Required Follow-ups", ""])
    for item in receipt.get("required_followups") or []:
        lines.append(f"- {item}")
    if not receipt.get("required_followups"):
        lines.append("- None.")

    lines.extend(["", "## Safety", ""])
    for key in sorted(safety):
        lines.append(f"- {key}: `{safety[key]}`")

    lines.extend(
        [
            "",
            "## Operator Note",
            "",
            "This receipt records a review decision only. It does not authorize live execution, broker access, production config writes, model promotion, or automatic learning-memory writes.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _load_source_artifact(path: str | Path) -> dict[str, Any]:
    source_path = Path(path)
    if not source_path.exists():
        return {
            "run_id": None,
            "mode": "missing_source_artifact",
            "errors": [f"Source artifact not found: {source_path}"],
        }

    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "run_id": None,
            "mode": "invalid_source_artifact",
            "errors": [repr(exc)],
        }

    return payload if isinstance(payload, dict) else {"run_id": None, "mode": "invalid_source_artifact", "errors": ["JSON is not an object"]}


def _source_decision(source: dict[str, Any]) -> str | None:
    if isinstance(source.get("post_dry_run_review"), dict):
        return source["post_dry_run_review"].get("decision")
    if isinstance(source.get("post_paper_simulation_review"), dict):
        return source["post_paper_simulation_review"].get("decision")
    if isinstance(source.get("decision"), dict):
        return source["decision"].get("decision")
    if isinstance(source.get("receipt"), dict):
        return source["receipt"].get("decision")
    if isinstance(source.get("summary"), dict):
        return str(source["summary"].get("ready_for_chief_review"))
    return None


def _source_verdict(source: dict[str, Any]) -> str | None:
    if isinstance(source.get("post_dry_run_review"), dict):
        return source["post_dry_run_review"].get("verdict")
    if isinstance(source.get("post_paper_simulation_review"), dict):
        return source["post_paper_simulation_review"].get("verdict")
    if isinstance(source.get("decision"), dict):
        return source["decision"].get("verdict")
    if isinstance(source.get("pipeline_report"), dict):
        return source["pipeline_report"].get("verdict")
    if isinstance(source.get("analytical_report"), dict):
        return source["analytical_report"].get("verdict")
    return None


def _source_summary(source: dict[str, Any]) -> dict[str, Any]:
    for key in ("post_dry_run_review", "post_paper_simulation_review"):
        if isinstance(source.get(key), dict):
            decision = source[key]
            return {
                "mode": source.get("mode"),
                "source_decision": decision.get("decision"),
                "source_verdict": decision.get("verdict"),
                "confidence": decision.get("confidence"),
                "data_quality_score": decision.get(
                    "data_quality_score"
                ),
                "source_sha256": None,
            }
    if isinstance(source.get("decision"), dict):
        decision = source["decision"]
        return {
            "mode": source.get("mode"),
            "source_decision": decision.get("decision"),
            "source_verdict": decision.get("verdict"),
            "confidence": decision.get("confidence"),
            "data_quality_score": decision.get("data_quality_score"),
        }
    if isinstance(source.get("summary"), dict):
        return {
            "mode": source.get("mode"),
            **source.get("summary", {}),
        }
    return {
        "mode": source.get("mode"),
        "run_id": source.get("run_id"),
        "errors": source.get("errors"),
    }


def _status_for(decision: ReviewDecisionType) -> ReviewDecisionStatus:
    return {
        "mark_reviewed": "reviewed",
        "needs_more_data": "needs_more_data",
        "reject": "rejected",
        "approve_dry_run": "dry_run_approved",
        "approve_paper_only_simulation": "paper_only_approved",
    }[decision]
