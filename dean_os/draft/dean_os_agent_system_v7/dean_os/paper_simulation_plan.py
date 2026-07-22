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
    object_fingerprint,
    receipt_lineage_issues,
)
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

PaperSimulationPlanStatus = Literal[
    "paper_simulation_plan_ready",
    "blocked_missing_receipt",
    "blocked_not_approved",
    "blocked_unsafe_receipt",
    "blocked_source_not_ready",
    "blocked_lineage_mismatch",
]

PaperSimulationStepType = Literal[
    "load_review_context",
    "validate_receipt",
    "validate_post_dry_run_review",
    "initialize_paper_sandbox",
    "run_paper_simulation_preview",
    "validate_risk_limits",
    "write_review_artifact",
]


class PaperSimulationStep(BaseModel):
    step_id: str = Field(default_factory=lambda: f"paper_sim_step_{uuid4().hex}")
    step_type: PaperSimulationStepType
    description: str
    command_preview: str | None = None
    required_inputs: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    guardrails: list[str] = Field(default_factory=list)
    can_execute_live: bool = False
    can_call_broker: bool = False
    can_write_production_config: bool = False


class PaperSimulationPlan(BaseModel):
    """A non-live paper simulation plan derived from a ReviewDecisionReceipt.

    This object is only a plan. It does not execute paper trades, call brokers,
    train/tune models, promote models, or write production config.
    """

    plan_id: str = Field(default_factory=lambda: f"paper_simulation_plan_{uuid4().hex}")
    created_at: str = Field(default_factory=utc_now_iso)
    status: PaperSimulationPlanStatus
    source_receipt_id: str | None = None
    source_receipt_path: str | None = None
    source_receipt_sha256: str | None = None
    source_receipt_fingerprint: str | None = None
    source_decision: str | None = None
    source_artifact_path: str | None = None
    source_artifact_decision: str | None = None
    source_artifact_verdict: str | None = None
    source_artifact_sha256: str | None = None
    lineage_verified: bool = False
    scope: Literal["paper_only", "blocked"] = "blocked"
    target: str = "paper_only_simulation_plan"
    steps: list[PaperSimulationStep] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    guardrails: list[str] = Field(default_factory=list)
    required_followups: list[str] = Field(default_factory=list)

    review_required: bool = True
    paper_simulation_plan_only: bool = True
    paper_simulation_executed: bool = False
    live_execution_allowed: bool = False
    broker_access_allowed: bool = False
    production_config_write_allowed: bool = False
    learning_memory_write_allowed: bool = False
    model_promotion_allowed: bool = False
    training_or_tuning_run_allowed: bool = False


class PaperSimulationPlanBuilder:
    """Builds paper simulation plans from review decision receipts."""

    def __init__(
        self,
        receipt_path: str | Path,
        output_dir: str | Path = "reports/dean_os/paper_simulation_plans",
    ):
        self.receipt_path = Path(receipt_path)
        self.output_dir = Path(output_dir)

    def build(self, save: bool = True) -> dict[str, Any]:
        receipt_payload = self._load_receipt()
        plan = build_paper_simulation_plan_from_receipt(
            receipt_payload,
            source_receipt_path=str(self.receipt_path),
            lineage_issues=receipt_lineage_issues(
                receipt_payload,
                receipt_path=self.receipt_path,
            ),
        )
        plan_payload = plan.model_dump(mode="json")
        payload = {
            "run_id": plan.plan_id,
            "mode": "paper_simulation_plan",
            "schema_version": PAPER_LIFECYCLE_SCHEMA_VERSION,
            "created_at": utc_now_iso(),
            "paper_simulation_plan": plan_payload,
            "plan_fingerprint": object_fingerprint(plan_payload),
            "source_receipt_summary": _receipt_summary(receipt_payload),
            "safety": {
                "review_only": True,
                "paper_simulation_plan_only": True,
                "paper_simulation_executed": False,
                "live_execution_allowed": False,
                "broker_access_allowed": False,
                "production_config_write_allowed": False,
                "learning_memory_write_allowed": False,
                "model_promotion_allowed": False,
                "training_or_tuning_run_allowed": False,
                "lineage_verified": plan.lineage_verified,
            },
        }

        if save:
            markdown = render_paper_simulation_plan_markdown(payload)
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=markdown,
                run_id=plan.plan_id,
            )
            payload["saved_paths"] = saved_paths

        return json_ready(payload)

    def _load_receipt(self) -> dict[str, Any]:
        if not self.receipt_path.exists():
            return {
                "mode": "missing_review_decision_receipt",
                "receipt": None,
                "errors": [f"Receipt artifact not found: {self.receipt_path}"],
            }

        try:
            payload = json.loads(self.receipt_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {
                "mode": "invalid_review_decision_receipt",
                "receipt": None,
                "errors": [repr(exc)],
            }

        if not isinstance(payload, dict):
            return {
                "mode": "invalid_review_decision_receipt",
                "receipt": None,
                "errors": ["Receipt JSON is not an object"],
            }
        return payload


def build_paper_simulation_plan_from_receipt(
    receipt_payload: dict[str, Any],
    source_receipt_path: str | None = None,
    lineage_issues: list[str] | None = None,
) -> PaperSimulationPlan:
    receipt = receipt_payload.get("receipt")
    if not isinstance(receipt, dict):
        return PaperSimulationPlan(
            status="blocked_missing_receipt",
            source_receipt_path=source_receipt_path,
            scope="blocked",
            reasons=["No valid review decision receipt was supplied."],
            risks=_base_risks(),
            guardrails=_base_guardrails(),
            required_followups=[
                "Create a ReviewDecisionReceipt with decision=approve_paper_only_simulation before planning paper simulation."
            ],
        )

    lineage_issues = list(lineage_issues or [])
    lineage = {
        "source_receipt_id": receipt.get("receipt_id"),
        "source_receipt_path": source_receipt_path,
        "source_receipt_sha256": file_sha256(source_receipt_path),
        "source_receipt_fingerprint": receipt_payload.get(
            "receipt_fingerprint"
        ),
        "source_artifact_path": receipt.get("source_artifact_path"),
        "source_artifact_sha256": receipt.get(
            "source_artifact_sha256"
        ),
    }
    if lineage_issues:
        return PaperSimulationPlan(
            status="blocked_lineage_mismatch",
            source_decision=receipt.get("decision"),
            source_artifact_decision=receipt.get("source_decision"),
            source_artifact_verdict=receipt.get("source_verdict"),
            scope="blocked",
            reasons=[
                "Receipt/source lineage is incomplete, expired, or changed.",
                *lineage_issues,
            ],
            risks=_base_risks(),
            guardrails=_base_guardrails(),
            required_followups=[
                "Create a fresh hash-bound review receipt from an unchanged "
                "post-dry-run review artifact."
            ],
            lineage_verified=False,
            **lineage,
        )

    dangerous = _unsafe_receipt_flags(receipt, receipt_payload)
    if dangerous:
        return PaperSimulationPlan(
            status="blocked_unsafe_receipt",
            source_receipt_id=receipt.get("receipt_id"),
            source_receipt_path=source_receipt_path,
            source_decision=receipt.get("decision"),
            source_artifact_path=receipt.get("source_artifact_path"),
            source_artifact_decision=receipt.get("source_decision"),
            source_artifact_verdict=receipt.get("source_verdict"),
            source_artifact_sha256=receipt.get("source_artifact_sha256"),
            source_receipt_sha256=lineage["source_receipt_sha256"],
            source_receipt_fingerprint=lineage[
                "source_receipt_fingerprint"
            ],
            scope="blocked",
            reasons=["Receipt contains unsafe flags.", *dangerous],
            risks=_base_risks(),
            guardrails=_base_guardrails(),
            required_followups=["Inspect and replace unsafe receipt before continuing."],
        )

    if receipt.get("decision") != "approve_paper_only_simulation" or receipt.get("scope") != "paper_only":
        return PaperSimulationPlan(
            status="blocked_not_approved",
            source_receipt_id=receipt.get("receipt_id"),
            source_receipt_path=source_receipt_path,
            source_decision=receipt.get("decision"),
            source_artifact_path=receipt.get("source_artifact_path"),
            source_artifact_decision=receipt.get("source_decision"),
            source_artifact_verdict=receipt.get("source_verdict"),
            source_artifact_sha256=receipt.get("source_artifact_sha256"),
            source_receipt_sha256=lineage["source_receipt_sha256"],
            source_receipt_fingerprint=lineage[
                "source_receipt_fingerprint"
            ],
            scope="blocked",
            reasons=[
                f"Receipt decision is {receipt.get('decision')!r}; paper simulation plan requires decision='approve_paper_only_simulation'.",
            ],
            risks=_base_risks(),
            guardrails=_base_guardrails(),
            required_followups=[
                "Ask reviewer for explicit approve_paper_only_simulation receipt if a non-live paper simulation plan is desired."
            ],
        )

    source_decision = str(receipt.get("source_decision") or "")
    source_verdict = str(receipt.get("source_verdict") or "")
    source_ready_decisions = {"ready_for_human_review", "ready_for_review", "mark_reviewed"}
    source_ready_verdicts = {"clear", "caution"}

    # If the source artifact exposes an explicit decision, it must be a ready decision.
    # Verdict is only a fallback for older artifacts without a structured decision.
    source_is_ready = (
        source_decision in source_ready_decisions
        or (not source_decision and source_verdict in source_ready_verdicts)
    )
    if not source_is_ready:
        return PaperSimulationPlan(
            status="blocked_source_not_ready",
            source_receipt_id=receipt.get("receipt_id"),
            source_receipt_path=source_receipt_path,
            source_decision=receipt.get("decision"),
            source_artifact_path=receipt.get("source_artifact_path"),
            source_artifact_decision=receipt.get("source_decision"),
            source_artifact_verdict=receipt.get("source_verdict"),
            source_artifact_sha256=receipt.get("source_artifact_sha256"),
            source_receipt_sha256=lineage["source_receipt_sha256"],
            source_receipt_fingerprint=lineage[
                "source_receipt_fingerprint"
            ],
            scope="blocked",
            reasons=[
                "Receipt approves paper-only simulation, but source artifact does not look ready for human-reviewed paper simulation.",
                f"source_decision={source_decision!r}, source_verdict={source_verdict!r}",
            ],
            risks=_base_risks(),
            guardrails=_base_guardrails(),
            required_followups=[
                "Use a receipt sourced from a ready PostDryRunReview / ChiefReview artifact before paper simulation planning."
            ],
        )

    return PaperSimulationPlan(
        status="paper_simulation_plan_ready",
        source_receipt_id=receipt.get("receipt_id"),
        source_receipt_path=source_receipt_path,
        source_decision=receipt.get("decision"),
        source_artifact_path=receipt.get("source_artifact_path"),
        source_artifact_decision=receipt.get("source_decision"),
        source_artifact_verdict=receipt.get("source_verdict"),
        source_artifact_sha256=receipt.get("source_artifact_sha256"),
        source_receipt_sha256=lineage["source_receipt_sha256"],
        source_receipt_fingerprint=lineage[
            "source_receipt_fingerprint"
        ],
        lineage_verified=True,
        scope="paper_only",
        target="approved_non_live_paper_simulation_plan",
        steps=_default_paper_steps(receipt),
        reasons=[
            "Receipt explicitly approved paper-only simulation scope.",
            "Plan remains non-live and isolated from broker/live execution.",
        ],
        risks=_base_risks(),
        guardrails=_base_guardrails(),
        required_followups=[
            "Run paper simulation only in isolated local/sandbox environment.",
            "Write paper simulation result artifacts only.",
            "Require a separate post-paper review before any promotion or future live consideration.",
        ],
    )


def render_paper_simulation_plan_markdown(payload: dict[str, Any]) -> str:
    plan = payload.get("paper_simulation_plan") or {}
    receipt = payload.get("source_receipt_summary") or {}
    lines = [
        "# DEAN-OS Paper Simulation Plan",
        "",
        f"- Plan ID: `{plan.get('plan_id')}`",
        f"- Status: `{plan.get('status')}`",
        f"- Scope: `{plan.get('scope')}`",
        f"- Source receipt: `{plan.get('source_receipt_id')}`",
        f"- Source receipt SHA256: `{plan.get('source_receipt_sha256')}`",
        f"- Lineage verified: `{plan.get('lineage_verified')}`",
        f"- Source decision: `{plan.get('source_decision')}`",
        f"- Source artifact decision: `{plan.get('source_artifact_decision')}`",
        f"- Source artifact verdict: `{plan.get('source_artifact_verdict')}`",
        "",
        "## Source Receipt Summary",
        "",
    ]

    for key, value in receipt.items():
        lines.append(f"- {key}: `{value}`")

    lines.extend(["", "## Reasons", ""])
    for item in plan.get("reasons") or []:
        lines.append(f"- {item}")

    lines.extend(["", "## Steps", ""])
    steps = plan.get("steps") or []
    if steps:
        lines.extend(["| Step | Type | Description | Command preview |", "|---|---|---|---|"])
        for idx, step in enumerate(steps, start=1):
            lines.append(
                "| {idx} | {type} | {desc} | `{cmd}` |".format(
                    idx=idx,
                    type=step.get("step_type"),
                    desc=str(step.get("description") or "").replace("|", "/"),
                    cmd=step.get("command_preview") or "",
                )
            )
    else:
        lines.append("- No steps because the paper simulation plan is blocked.")

    lines.extend(["", "## Guardrails", ""])
    for item in plan.get("guardrails") or []:
        lines.append(f"- {item}")

    lines.extend(["", "## Required Follow-ups", ""])
    for item in plan.get("required_followups") or []:
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
            "This is a plan only. It does not execute a paper simulation and cannot interact with a broker or live account.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _default_paper_steps(receipt: dict[str, Any]) -> list[PaperSimulationStep]:
    source_path = receipt.get("source_artifact_path") or "reports/dean_os/post_dry_run_review/latest.json"
    return [
        PaperSimulationStep(
            step_type="load_review_context",
            description="Load post-dry-run review, dry-run result, and receipt artifacts.",
            command_preview=f"read-only load {source_path}",
            required_inputs=[str(source_path), "review_decision_receipt"],
            expected_outputs=["paper_context_snapshot"],
            guardrails=["read_only", "no_network_required"],
        ),
        PaperSimulationStep(
            step_type="validate_receipt",
            description="Verify the receipt allows paper-only simulation and all live/prod flags remain false.",
            command_preview="validate receipt --scope paper_only --no-live --no-broker --no-production-write",
            required_inputs=["review_decision_receipt"],
            expected_outputs=["receipt_guardrail_validation"],
            guardrails=_base_guardrails(),
        ),
        PaperSimulationStep(
            step_type="validate_post_dry_run_review",
            description="Validate that the source post-dry-run review is ready for human-reviewed next step.",
            command_preview="validate post-dry-run-review --ready-for-human-review",
            required_inputs=["post_dry_run_review"],
            expected_outputs=["post_dry_run_validation_report"],
            guardrails=["human_review_required", "no_auto_approval"],
        ),
        PaperSimulationStep(
            step_type="initialize_paper_sandbox",
            description="Initialize isolated paper-only sandbox state with no broker/live credentials.",
            command_preview="paper-sim init --local-sandbox --no-broker --no-live",
            required_inputs=["paper_context_snapshot"],
            expected_outputs=["paper_sandbox_state"],
            guardrails=["no_broker_access", "no_live_execution", "local_only"],
        ),
        PaperSimulationStep(
            step_type="run_paper_simulation_preview",
            description="Run a preview paper simulation using saved artifacts and bounded risk settings.",
            command_preview="paper-sim preview --read-only-inputs --write-review-artifact-only",
            required_inputs=["paper_sandbox_state", "paper_context_snapshot"],
            expected_outputs=["paper_simulation_preview_report"],
            guardrails=["no_real_orders", "no_config_write", "no_learning_write"],
        ),
        PaperSimulationStep(
            step_type="validate_risk_limits",
            description="Validate exposure/drawdown/order-count constraints in the paper simulation preview.",
            command_preview="paper-sim validate-risk --max-exposure-review --max-drawdown-review",
            required_inputs=["paper_simulation_preview_report"],
            expected_outputs=["paper_simulation_risk_report"],
            guardrails=["risk_review_required", "no_limit_relaxation"],
        ),
        PaperSimulationStep(
            step_type="write_review_artifact",
            description="Write paper simulation plan/result outputs as JSON/Markdown review artifacts only.",
            command_preview="write reports/dean_os/paper_simulation_results/latest.json latest.md",
            required_inputs=["paper_simulation_preview_report", "paper_simulation_risk_report"],
            expected_outputs=["paper_simulation_result_artifact"],
            guardrails=["atomic_local_write", "no_learning_write", "no_production_config_write"],
        ),
    ]


def _base_guardrails() -> list[str]:
    return [
        "paper_only",
        "review_only",
        "no_live_execution",
        "no_broker_access",
        "no_production_config_write",
        "no_learning_memory_write",
        "no_model_promotion",
        "human_review_required_after_paper_simulation",
    ]


def _base_risks() -> list[str]:
    return [
        "Paper simulation success does not authorize live execution.",
        "Paper simulation is not production approval.",
        "A separate post-paper review is required before any later promotion workflow.",
        "Broker/live credentials must not be available to the paper simulation path.",
    ]


def _unsafe_receipt_flags(receipt: dict[str, Any], payload: dict[str, Any]) -> list[str]:
    dangerous_flags = {
        "live_execution_allowed",
        "broker_access_allowed",
        "production_config_write_allowed",
        "learning_memory_write_allowed",
        "model_promotion_allowed",
    }
    messages: list[str] = []
    safety = dict(payload.get("safety") or {})
    for flag in dangerous_flags:
        if receipt.get(flag) is True:
            messages.append(f"receipt.{flag}=true")
        if safety.get(flag) is True:
            messages.append(f"safety.{flag}=true")
    return messages


def _receipt_summary(payload: dict[str, Any]) -> dict[str, Any]:
    receipt = payload.get("receipt") if isinstance(payload.get("receipt"), dict) else {}
    source = payload.get("source_summary") if isinstance(payload.get("source_summary"), dict) else {}
    return {
        "receipt_id": receipt.get("receipt_id"),
        "decision": receipt.get("decision"),
        "status": receipt.get("status"),
        "scope": receipt.get("scope"),
        "reviewer": receipt.get("reviewer"),
        "source_decision": receipt.get("source_decision") or source.get("source_decision"),
        "source_verdict": receipt.get("source_verdict") or source.get("source_verdict"),
    }
