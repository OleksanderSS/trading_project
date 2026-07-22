from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

DryRunPlanStatus = Literal[
    "dry_run_plan_ready",
    "blocked_missing_receipt",
    "blocked_not_approved",
    "blocked_unsafe_receipt",
]

DryRunStepType = Literal[
    "load_context",
    "validate_guardrails",
    "run_replay",
    "run_walk_forward_dry_run",
    "compare_outputs",
    "write_review_artifact",
]


class DryRunStep(BaseModel):
    step_id: str = Field(default_factory=lambda: f"dry_run_step_{uuid4().hex}")
    step_type: DryRunStepType
    description: str
    command_preview: str | None = None
    required_inputs: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    guardrails: list[str] = Field(default_factory=list)
    can_execute_live: bool = False
    can_write_production_config: bool = False


class DryRunExecutionPlan(BaseModel):
    """A non-live dry-run/simulation plan derived from a ReviewDecisionReceipt.

    This object is only a plan. It does not execute training, tuning, trading,
    broker calls, model promotion, or production config writes.
    """

    plan_id: str = Field(default_factory=lambda: f"dry_run_plan_{uuid4().hex}")
    created_at: str = Field(default_factory=utc_now_iso)
    status: DryRunPlanStatus
    source_receipt_id: str | None = None
    source_receipt_path: str | None = None
    source_decision: str | None = None
    scope: Literal["dry_run", "blocked"] = "blocked"
    target: str = "review_only_dry_run"
    steps: list[DryRunStep] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    guardrails: list[str] = Field(default_factory=list)
    required_followups: list[str] = Field(default_factory=list)

    review_required: bool = True
    live_execution_allowed: bool = False
    broker_access_allowed: bool = False
    production_config_write_allowed: bool = False
    learning_memory_write_allowed: bool = False
    model_promotion_allowed: bool = False
    training_or_tuning_run_allowed: bool = False


class DryRunExecutionPlanBuilder:
    """Builds dry-run plans from review decision receipts."""

    def __init__(
        self,
        receipt_path: str | Path,
        output_dir: str | Path = "reports/dean_os/dry_run_plans",
    ):
        self.receipt_path = Path(receipt_path)
        self.output_dir = Path(output_dir)

    def build(self, save: bool = True) -> dict[str, Any]:
        receipt_payload = self._load_receipt()
        plan = build_dry_run_plan_from_receipt(receipt_payload, source_receipt_path=str(self.receipt_path))
        payload = {
            "run_id": plan.plan_id,
            "mode": "dry_run_execution_plan",
            "created_at": utc_now_iso(),
            "dry_run_plan": plan.model_dump(mode="json"),
            "source_receipt_summary": _receipt_summary(receipt_payload),
            "safety": {
                "review_only": True,
                "dry_run_plan_only": True,
                "dry_run_executed": False,
                "live_execution_allowed": False,
                "broker_access_allowed": False,
                "production_config_write_allowed": False,
                "learning_memory_write_allowed": False,
                "model_promotion_allowed": False,
                "training_or_tuning_run_allowed": False,
            },
        }

        if save:
            markdown = render_dry_run_plan_markdown(payload)
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


def build_dry_run_plan_from_receipt(receipt_payload: dict[str, Any], source_receipt_path: str | None = None) -> DryRunExecutionPlan:
    receipt = receipt_payload.get("receipt")
    if not isinstance(receipt, dict):
        return DryRunExecutionPlan(
            status="blocked_missing_receipt",
            source_receipt_path=source_receipt_path,
            scope="blocked",
            reasons=["No valid review decision receipt was supplied."],
            risks=_base_risks(),
            guardrails=_base_guardrails(),
            required_followups=["Create a ReviewDecisionReceipt with decision=approve_dry_run before planning a dry-run."],
        )

    dangerous = _unsafe_receipt_flags(receipt, receipt_payload)
    if dangerous:
        return DryRunExecutionPlan(
            status="blocked_unsafe_receipt",
            source_receipt_id=receipt.get("receipt_id"),
            source_receipt_path=source_receipt_path,
            source_decision=receipt.get("decision"),
            scope="blocked",
            reasons=["Receipt contains unsafe flags.", *dangerous],
            risks=_base_risks(),
            guardrails=_base_guardrails(),
            required_followups=["Inspect and replace unsafe receipt before continuing."],
        )

    if receipt.get("decision") != "approve_dry_run" or receipt.get("scope") != "dry_run":
        return DryRunExecutionPlan(
            status="blocked_not_approved",
            source_receipt_id=receipt.get("receipt_id"),
            source_receipt_path=source_receipt_path,
            source_decision=receipt.get("decision"),
            scope="blocked",
            reasons=[
                f"Receipt decision is {receipt.get('decision')!r}; dry-run plan requires decision='approve_dry_run'.",
            ],
            risks=_base_risks(),
            guardrails=_base_guardrails(),
            required_followups=[
                "Ask reviewer for explicit approve_dry_run receipt if a non-live simulation plan is desired."
            ],
        )

    steps = _default_dry_run_steps(receipt)
    return DryRunExecutionPlan(
        status="dry_run_plan_ready",
        source_receipt_id=receipt.get("receipt_id"),
        source_receipt_path=source_receipt_path,
        source_decision=receipt.get("decision"),
        scope="dry_run",
        target="approved_review_only_dry_run",
        steps=steps,
        reasons=[
            "Receipt explicitly approved dry-run scope.",
            "Plan remains non-live and review-only.",
        ],
        risks=_base_risks(),
        guardrails=_base_guardrails(),
        required_followups=[
            "Run dry-run only in isolated review environment.",
            "Write dry-run results as review artifacts only.",
            "Require a separate human review before any paper simulation or promotion step.",
        ],
        review_required=True,
        live_execution_allowed=False,
        broker_access_allowed=False,
        production_config_write_allowed=False,
        learning_memory_write_allowed=False,
        model_promotion_allowed=False,
        training_or_tuning_run_allowed=False,
    )


def render_dry_run_plan_markdown(payload: dict[str, Any]) -> str:
    plan = payload.get("dry_run_plan") or {}
    receipt = payload.get("source_receipt_summary") or {}
    lines = [
        "# DEAN-OS Dry Run Execution Plan",
        "",
        f"- Plan ID: `{plan.get('plan_id')}`",
        f"- Status: `{plan.get('status')}`",
        f"- Scope: `{plan.get('scope')}`",
        f"- Source receipt: `{plan.get('source_receipt_id')}`",
        f"- Source decision: `{plan.get('source_decision')}`",
        f"- Target: `{plan.get('target')}`",
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
        lines.append("- No steps because the dry-run plan is blocked.")

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
            "This is a plan only. It does not execute a dry-run. A separate tool/agent may later consume this plan, but only in an isolated, non-live environment.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _default_dry_run_steps(receipt: dict[str, Any]) -> list[DryRunStep]:
    source_path = receipt.get("source_artifact_path") or "reports/dean_os/chief_review_index/latest.json"
    return [
        DryRunStep(
            step_type="load_context",
            description="Load source review artifacts and the reviewed context snapshot.",
            command_preview=f"read-only load {source_path}",
            required_inputs=[str(source_path)],
            expected_outputs=["context_snapshot"],
            guardrails=["read_only", "no_network_required"],
        ),
        DryRunStep(
            step_type="validate_guardrails",
            description="Verify the receipt allows dry-run only and all live/prod flags remain false.",
            command_preview="validate receipt --scope dry_run --no-live --no-production-write",
            required_inputs=["review_decision_receipt"],
            expected_outputs=["guardrail_validation_report"],
            guardrails=_base_guardrails(),
        ),
        DryRunStep(
            step_type="run_replay",
            description="Replay analyst/tuning decisions against saved artifacts without side effects.",
            command_preview="dry-run replay --read-only --no-broker --no-production-write",
            required_inputs=["review_index", "chief_review_index", "review_decision_receipt"],
            expected_outputs=["dry_run_replay_report"],
            guardrails=["read_only", "no_broker", "no_config_write"],
        ),
        DryRunStep(
            step_type="compare_outputs",
            description="Compare dry-run outputs against expected guardrails and artifact recommendations.",
            command_preview="compare dry-run outputs --against review_index --against chief_review",
            required_inputs=["dry_run_replay_report", "review_index", "chief_review_index"],
            expected_outputs=["dry_run_comparison_report"],
            guardrails=["no_promotion", "human_review_required"],
        ),
        DryRunStep(
            step_type="write_review_artifact",
            description="Write dry-run results as JSON/Markdown review artifacts only.",
            command_preview="write reports/dean_os/dry_run_results/latest.json latest.md",
            required_inputs=["dry_run_comparison_report"],
            expected_outputs=["dry_run_result_artifact"],
            guardrails=["atomic_local_write", "no_learning_write", "no_production_config_write"],
        ),
    ]


def _base_guardrails() -> list[str]:
    return [
        "review_only",
        "dry_run_only",
        "no_live_execution",
        "no_broker_access",
        "no_production_config_write",
        "no_learning_memory_write",
        "no_model_promotion",
        "human_review_required_after_dry_run",
    ]


def _base_risks() -> list[str]:
    return [
        "Dry-run artifacts must not be interpreted as production approval.",
        "A separate human review is required before any paper-only simulation or model promotion workflow.",
        "Live execution remains forbidden.",
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
